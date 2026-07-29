#include "ascend/include/CVSplitScheduling/CVSplitScheduling.h"
#include "ascend/include/CVSplitScheduling/CrossScopeTransfers.h"
#include "ascend/include/CVSplitScheduling/DependencyScheduler.h"
#include "ascend/include/CVSplitScheduling/PreCheck.h"
#include "ascend/include/CVSplitScheduling/QStaging.h"
#include "ascend/include/CVSplitScheduling/ScopeSeparation.h"
#include "ascend/include/CVSplitScheduling/UnrollOrigin.h"
#include "ascend/include/CVSplitScheduling/UnfusePVMatmuls.h"
#include "ascend/include/CVSplitScheduling/classifyAllOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <functional>

#define DEBUG_TYPE "cv-split-scheduling"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CVSPLITSCHEDULING
#include "ascend/include/CVSplitScheduling/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

// ============================================================================
// CV-Split Scheduling
// ----------------------------------------------------------------------------
// Splits the innermost loop of a fused kernel into two co-running engine scopes
// — a CUBE scope (matmul / fixpipe) and a VECTOR scope (elementwise / softmax) —
// with explicit cross-engine buffers and synchronization, so the Ascend cube
// and vector units overlap instead of running serially.
//
// Pipeline (driven by CVSplitSchedulingPass::processFunction):
//   1.  findInnermostLoop            locate the loop to split
//   1b. hasStoresInBody              bail if the body already stores (not fusable)
//   2.  loopUnrollByFactor           unroll by `unroll-factor` to expose ILP
//   3.  classifyAllOps               tag each op CUBE or VECTOR (matmul-seeded,
//                                    data-feeders pulled into CUBE, rest VECTOR)
//   4-7 DependencyScheduler          graph -> BFS levels -> reorder so
//                                    same-engine work is contiguous
//   7.5 unfusePVMatmuls              undo matmul(p,v,acc) fusion that entangles
//                                    the engines
//   8.  insertCrossScopeTransfers    materialize C->V (fixpipe->UB) and V->C
//                                    (NZ pack->L1) buffers + sync_block_set/wait
//   9.  createScopeSeparation        clone CUBE ops into a CUBE scope, wrap the
//                                    loop+epilogue in a VECTOR scope, strip the
//                                    wrong-engine ops from each, then ROW_SPLIT
//                                    re-tile the VECTOR scope across both veccores
//
// Generality: the pass is engine-pattern driven, not kernel-name driven — it
// keys off op semantics (matmul == CUBE, float elementwise == VECTOR) and bails
// out cleanly (leaving the IR untouched) whenever an assumption does not hold:
// no innermost loop, stores already present, unroll-factor <= 1, no CUBE ops, or
// CUBE/VECTOR work that the level scheduler finds entangled. Flash-Attention is
// the validated driver kernel; other fused cube+vector loops that satisfy the
// same structural contract are handled by the same path, and anything else
// falls through unmodified.
// ============================================================================

namespace {

using cv_split::kUnrollOriginIdAttrName;

// FIXME: remove before PR.
static void dumpOriginTagIR(ModuleOp module, StringRef fileName) {
  llvm::SmallString<256> dumpDir(__FILE__);
  llvm::sys::path::remove_filename(dumpDir);
  llvm::sys::path::append(dumpDir, "tmp_dbg");

  if (std::error_code ec = llvm::sys::fs::create_directories(dumpDir)) {
    llvm::errs() << "[cv-split] Failed to create debug dump directory "
                 << dumpDir << ": " << ec.message() << "\n";
    return;
  }

  llvm::SmallString<256> outputPath(dumpDir);
  llvm::sys::path::append(outputPath, fileName);

  std::error_code ec;
  llvm::raw_fd_ostream output(outputPath, ec);
  if (ec) {
    llvm::errs() << "[cv-split] Failed to open debug dump " << outputPath
                 << ": " << ec.message() << "\n";
    return;
  }

  module.print(output);
  output << '\n';
}

static void tagUnrollOriginIds(scf::ForOp loop) {
  Builder builder(loop.getContext());
  int64_t originId = 0;
  for (Operation &op : *loop.getBody()) {
    if (isa<scf::YieldOp>(op))
      continue;
    op.setAttr(kUnrollOriginIdAttrName,
               builder.getI64IntegerAttr(originId++));
  }
}

static void removeUnrollOriginIdAttrs(Operation *root) {
  root->walk(
      [](Operation *op) { op->removeAttr(kUnrollOriginIdAttrName); });
}

static void commitModuleClone(ModuleOp destination, ModuleOp source) {
  Operation *destinationOp = destination.getOperation();
  Operation *sourceOp = source.getOperation();

  destinationOp->setLoc(sourceOp->getLoc());
  destinationOp->setAttrs(sourceOp->getAttrs());
  if (destinationOp->getPropertiesStorageSize() != 0)
    destinationOp->copyProperties(sourceOp->getPropertiesStorage());
  destination.getBodyRegion().takeBody(source.getBodyRegion());
}

static void commitFunctionClone(func::FuncOp destination,
                                func::FuncOp source) {
  Operation *destinationOp = destination.getOperation();
  Operation *sourceOp = source.getOperation();

  destinationOp->setLoc(sourceOp->getLoc());
  destinationOp->setAttrs(sourceOp->getAttrs());
  if (destinationOp->getPropertiesStorageSize() != 0)
    destinationOp->copyProperties(sourceOp->getPropertiesStorage());
  destination.getBody().takeBody(source.getBody());
}

struct FunctionBackup {
  explicit FunctionBackup(func::FuncOp function)
      : function(function), backup(function.clone()) {}

  func::FuncOp function;
  OwningOpRef<func::FuncOp> backup;
};

struct CandidateState {
  FunctionBackup *functionBackup;
  scf::ForOp loop;
};

static void restoreFunction(FunctionBackup &state) {
  commitFunctionClone(state.function, *state.backup);
}

static void restoreAndRefreshFunctionBackup(FunctionBackup &state) {
  restoreFunction(state);
  state.backup = OwningOpRef<func::FuncOp>(state.function.clone());
}
// ============================================================================
// Pass entry point
// ============================================================================
class CVSplitSchedulingPass
    : public ::impl::CVSplitSchedulingBase<CVSplitSchedulingPass> {
public:
  explicit CVSplitSchedulingPass(const CVSplitSchedulingOptions &options) {
    this->compileOn91095 = options.compileOn91095;
    this->unrollFactor = options.unrollFactor;
  }

  void runOnOperation() override {
    if (!compileOn91095) {
      llvm::errs() << "[cv-split] Not A5 target, skipping\n";
      return;
    }

    ModuleOp moduleOp = getOperation();
    llvm::errs() << "\n[cv-split] ============================\n"
                 << "[cv-split]  CVSplitScheduling START\n"
                 << "[cv-split]  unrollFactor=" << unrollFactor << "\n"
                 << "[cv-split] ============================\n\n";

    // Dump IR BEFORE the pass
    llvm::errs() << "[cv-split] === IR DUMP BEFORE CV-SPLIT PASS ===\n";
    moduleOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END IR DUMP BEFORE ===\n\n";

    // Run the transformation transactionally on one clone of the input module.
    // Each function has its own backup, so a failed candidate can be restored
    // without discarding successful candidates in other functions.
    OwningOpRef<ModuleOp> transformedModule = moduleOp.clone();
    SmallVector<FunctionBackup> functionBackups;
    for (func::FuncOp funcOp : transformedModule->getOps<func::FuncOp>())
      functionBackups.emplace_back(funcOp);

    SmallVector<CandidateState> candidates =
        prepareCandidates(functionBackups);
    llvm::errs() << "[cv-split] Functions: " << functionBackups.size()
                 << ", prepared candidates: " << candidates.size() << "\n";
    if (candidates.empty()) {
      llvm::errs() << "[cv-split] No candidate found; keeping original IR\n";
      return;
    }

    // Stage 3: DCVP remains unchanged and classifies the whole working module
    // exactly once. Non-candidate functions are restored immediately afterward
    // so classifier-side rewrites cannot leak into them.
    if (failed(cv_split::runDCVPClassifier(*transformedModule))) {
      llvm::errs() << "[cv-split] DCVP classification failed; keeping original "
                      "IR\n";
      return;
    }
    restoreNonCandidates(functionBackups, candidates);

    // Stages 4 onward: finish every prepared candidate independently. A failed
    // function is restored from its original backup and processing continues.
    if (!processCandidates(candidates)) {
      llvm::errs() << "[cv-split] No candidate transformed; keeping original "
                      "IR\n";
      return;
    }

    // Safety cleanup for functions that returned before the normal Stage 8
    // cleanup point.
    removeUnrollOriginIdAttrs(*transformedModule);
    cv_split::removeDCVPClassificationAttrs(*transformedModule);

    if (failed(verify(*transformedModule))) {
      llvm::errs() << "[cv-split] Transformed IR failed verification; keeping "
                      "original IR\n";
      return;
    }

    commitModuleClone(moduleOp, *transformedModule);

    llvm::errs() << "\n[cv-split] ============================\n"
                 << "[cv-split]  CVSplitScheduling END\n"
                 << "[cv-split] ============================\n\n";

    // Dump IR AFTER the pass
    llvm::errs() << "[cv-split] === IR DUMP AFTER CV-SPLIT PASS ===\n";
    moduleOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END IR DUMP AFTER ===\n\n";
  }

private:
  SmallVector<CandidateState>
  prepareCandidates(MutableArrayRef<FunctionBackup> functionBackups) {
    SmallVector<CandidateState> candidates;
    for (FunctionBackup &state : functionBackups) {
      llvm::errs() << "[cv-split] Function: " << state.function.getName()
                   << "\n";
      FailureOr<scf::ForOp> preCheckResult =
          preCheckCVSplitScheduling(state.function, unrollFactor);
      if (failed(preCheckResult)) {
        llvm::errs() << "[cv-split] Pre-check rejected function, skip\n";
        continue;
      }

      scf::ForOp candidateLoop = *preCheckResult;
      llvm::errs() << "[cv-split] Pre-check accepted candidate loop\n";
      if (failed(unrollCandidateLoop(state.function, candidateLoop))) {
        llvm::errs() << "[cv-split] Candidate preparation failed; trying "
                        "next function\n";
        restoreAndRefreshFunctionBackup(state);
        continue;
      }

      candidates.push_back({&state, candidateLoop});
    }
    return candidates;
  }

  static void
  restoreNonCandidates(MutableArrayRef<FunctionBackup> functionBackups,
                       ArrayRef<CandidateState> candidates) {
    llvm::DenseSet<Operation *> candidateFunctions;
    for (const CandidateState &candidate : candidates)
      candidateFunctions.insert(candidate.functionBackup->function);

    for (FunctionBackup &state : functionBackups)
      if (!candidateFunctions.contains(state.function))
        restoreFunction(state);
  }

  bool processCandidates(MutableArrayRef<CandidateState> candidates) {
    bool transformedAnyCandidate = false;
    for (CandidateState &candidate : candidates) {
      FunctionBackup &state = *candidate.functionBackup;
      if (failed(processFunction(state.function, candidate.loop)) ||
          failed(verify(state.function))) {
        llvm::errs() << "[cv-split] Candidate failed; restoring function and "
                        "trying next function\n";
        restoreFunction(state);
        continue;
      }
      transformedAnyCandidate = true;
    }
    return transformedAnyCandidate;
  }

  LogicalResult unrollCandidateLoop(func::FuncOp funcOp, scf::ForOp loop) {
    ModuleOp module = funcOp->getParentOfType<ModuleOp>();
    dumpOriginTagIR(module, "before_tag.mlir");
    tagUnrollOriginIds(loop);
    dumpOriginTagIR(module, "after_tag.mlir");

    // Stage 2: Unroll the innermost loop
    LogicalResult unrollResult = loopUnrollByFactor(loop, unrollFactor);
    if (failed(unrollResult)) {
      llvm::errs() << "[cv-split] Unroll failed, bail\n";
      return failure();
    }
    llvm::errs() << "[cv-split] Unrolled by " << unrollFactor << "\n";
    return success();
  }

  LogicalResult processFunction(func::FuncOp funcOp, scf::ForOp loop) {
    Block *body = loop.getBody();

    // Stage 3: Import the classifications stamped by the single module-level
    // DCVP classifier invocation in runOnOperation().
    FailureOr<cv_split::Classification> classificationResult =
        cv_split::readDCVPClassification(body);
    if (failed(classificationResult)) {
      llvm::errs() << "[cv-split] Failed to read DCVP classification, bail\n";
      return failure();
    }
    cv_split::Classification classification =
        std::move(*classificationResult);
    if (!cv_split::checkCoreClassifications(body, classification)) {
      llvm::errs() << "[cv-split] Loop must contain both CUBE and VECTOR ops, "
                      "skip\n";
      return failure();
    }

    // Stages 4-7: build the dependency graph, assign BFS levels, verify the
    // CUBE/VECTOR work is cleanly separable, and reorder the body by level.
    cv_split::DependencyScheduler scheduler;
    if (failed(scheduler.run(body, classification)))
      return failure();

    // Dump IR before scope separation
    llvm::errs() << "[cv-split] === IR BEFORE SCOPE SEPARATION ===\n";
    funcOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END IR BEFORE ===\n\n";

    // Stage 7.5: Unfuse PV matmuls (split matmul(p,v,acc*alpha) into pv + addf)
    if (failed(cv_split::unfusePVMatmuls(body, classification)))
      return failure();

    // Stage 8: Insert cross-scope transfers (BEFORE scope separation)
    llvm::errs() << "[cv-split] === Stage 8: cross-scope transfers ===\n";
    FailureOr<cv_split::CrossScopeTransferInfo> transferInfo =
        cv_split::insertCrossScopeTransfers(loop, classification);
    if (failed(transferInfo)) {
      return failure();
    }
    // Origin IDs are temporary unroll-lineage metadata. Transfer grouping is
    // their final consumer, so do not expose them to scope/backend passes.
    removeUnrollOriginIdAttrs(funcOp);
    llvm::errs() << "[cv-split] Stage 8 complete\n";

    // Dump IR after transfers, before scope separation
    llvm::errs() << "[cv-split] === IR AFTER TRANSFERS ===\n";
    funcOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END IR AFTER TRANSFERS ===\n\n";

    // Stage 9: Scope separation (like DynamicCVPipeline/SeparateCVScope)
    llvm::errs() << "[cv-split] === Stage 9: scope separation ===\n";
    if (failed(cv_split::createScopeSeparation(funcOp, loop, classification,
                                               *transferInfo))) {
      return failure();
    }
    llvm::errs() << "[cv-split] Stage 9 complete\n";

    // Stage 11.5: bind the loop-invariant matmul LHS (Q) into a cbuf buffer so
    // the QK matmul reads an aligned NZ L1 operand (matches the manual kernel
    // and avoids the misaligned implicit GM/UB->L1 stage of a plain memref).
    if (failed(cv_split::bindLoopInvariantMatmulLhsToCbuf(funcOp)))
      return failure();

    // Stage 10: Ensure function has mix_mode attribute (it should already)
    // Note: do NOT add hivm.func_core_type=MIX — that triggers SplitMixKernel
    // which conflicts with our already-scoped IR. The scope::ScopeOp attrs +
    // mix_mode="mix" are sufficient for BiShengIR to handle the scopes.
    if (!funcOp->hasAttr("mix_mode"))
      funcOp->setAttr("mix_mode", StringAttr::get(funcOp.getContext(), "mix"));
    llvm::errs() << "[cv-split] Function attributes set on " << funcOp.getName() << "\n";

    // Stage 11: Set module attribute to disable auto-tiling
    // Without this, BiShengIR's auto-tile pass creates invalid pointer_casts
    // inside our scoped loops (they're not IsolatedFromAbove).
    if (auto moduleOp = funcOp->getParentOfType<ModuleOp>()) {
      moduleOp->setAttr("hivm.disable_auto_tile_and_bind_subblock",
                        UnitAttr::get(funcOp.getContext()));
    }

    // Dump function IR after scope separation
    llvm::errs() << "[cv-split] === FUNCTION IR AFTER SCOPE SEPARATION ===\n";
    funcOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END FUNCTION IR ===\n";
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCVSplitSchedulingPass(
    const CVSplitSchedulingOptions &options) {
  return std::make_unique<CVSplitSchedulingPass>(options);
}
