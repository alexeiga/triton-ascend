#include "ascend/include/CVSplitScheduling/classifyAllOps.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::cv_split {

void setOpEngineTypeAttr(Operation *op, EngineType engineType) {
  StringRef coreType =
      engineType == EngineType::CUBE ? "CUBE" : "VECTOR";
  op->setAttr(CVPipeline::kCoreType,
              StringAttr::get(op->getContext(), coreType));
}

// FIXME: remove before PR.
static void dumpClassifierIR(ModuleOp module, StringRef fileName) {
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

FailureOr<Classification> classifyAllOpsWithDCVP(ModuleOp module,
                                                 Block *body) {
  PassManager pm(module.getContext(), module.getOperationName());
  pm.addPass(createOpClassifierPass());

  dumpClassifierIR(module, "before_dcvp_op_classifier.mlir");
  if (failed(pm.run(module)))
    return failure();
  dumpClassifierIR(module, "after_dcvp_op_classifier.mlir");

  Classification classification;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(op))
      continue;

    switch (CVPipeline::getOpCoreType(&op)) {
    case CVPipeline::CUBE_ONLY:
      classification[&op] = EngineType::CUBE;
      break;
    case CVPipeline::VECTOR_ONLY:
      classification[&op] = EngineType::VECTOR;
      break;
    default:
      op.emitError("DCVP did not produce a single-core classification");
      return failure();
    }
  }

  return classification;
}

bool checkCoreClassifications(Block *body,
                              const Classification &classification) {
  int nCube = 0;
  int nVector = 0;
  for (const auto &entry : classification) {
    if (entry.second == EngineType::CUBE)
      ++nCube;
    else
      ++nVector;
  }

  llvm::errs() << "[cv-split] Classification: " << nCube << "C "
               << nVector << "V\n";
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(op))
      continue;

    auto it = classification.find(&op);
    llvm::errs() << "[cv-split]   ";
    if (it == classification.end())
      llvm::errs() << "??";
    else
      llvm::errs() << (it->second == EngineType::CUBE ? "CUBE" : "VECTOR");
    llvm::errs() << " " << op.getName() << "\n";
  }

  return nCube > 0 && nVector > 0;
}

void removeDCVPClassificationAttrs(ModuleOp module) {
  module.walk([](Operation *op) {
    op->removeAttr(CVPipeline::kCoreType);
  });
}

} // namespace mlir::triton::cv_split
