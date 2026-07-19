#include "ascend/include/CVSplitScheduling/DependencyScheduler.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;

namespace mlir::triton::cv_split {

// ============================================================================
// Stage 4: Dependency graph
// ============================================================================
static void buildDependencyGraph(
    Block *body,
    DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
    DenseMap<Operation *, SmallVector<Operation *>> &successors) {
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    for (Value operand : op.getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != body || isa<scf::YieldOp>(defOp))
        continue;
      predecessors[&op].push_back(defOp);
      successors[defOp].push_back(&op);
    }
  }

  // Add memory dependency edges: memref.copy → bufferization.to_tensor
  // memref.copy writes to an alloc via side effect (no SSA result).
  // to_tensor reads from the same alloc. Without this edge, dependency leveling
  // can place to_tensor BEFORE copy, causing reads of uninitialized data.
  for (Operation &op : *body) {
    auto copyOp = dyn_cast<memref::CopyOp>(&op);
    if (!copyOp)
      continue;
    Value dst = copyOp.getTarget();
    for (Operation *user : dst.getUsers()) {
      if (user == &op || user->getBlock() != body)
        continue;
      if (isa<bufferization::ToTensorOp>(user)) {
        predecessors[user].push_back(&op);
        successors[&op].push_back(user);
      }
    }
  }

  // Similarly: memref.copy also depends on reinterpret_cast of its SOURCE.
  // The source reinterpret_cast depends on address arithmetic that computes
  // the GM offset. Ensure copy is ordered after its source address is ready.
  // (This is already captured by SSA edges since copy USES the source value.)
}

static bool validateDependencyGraph(
    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors) {
  for (const auto &entry : predecessors) {
    if (entry.second.empty()) {
      llvm::errs() << "[cv-split] Invalid empty predecessor entry for "
                   << entry.first->getName() << ", bail\n";
      return false;
    }
  }
  return true;
}

// ============================================================================
// Stage 5: Dependency level assignment
// ============================================================================
static SmallVector<Operation *> collectRoots(
    Block *body,
    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors) {
  SmallVector<Operation *> roots;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    auto it = predecessors.find(&op);
    if (it == predecessors.end()) {
      roots.push_back(&op);
      llvm::errs() << "[cv-split] Root: " << op.getName() << "\n";
    }
  }
  return roots;
}

static int assignDependencyLevels(
    Block *body,
    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
    const SmallVector<Operation *> &roots,
    DenseMap<Operation *, int> &levels) {
  for (auto *root : roots)
    levels[root] = 0;

  int maxLevel = 0;

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(&op))
        continue;

      auto it = predecessors.find(&op);
      if (it == predecessors.end())
        continue;

      int requiredLevel = 0;
      bool allPredecessorsReady = true;
      for (auto *pred : it->second) {
        auto predIt = levels.find(pred);
        if (predIt == levels.end()) {
          allPredecessorsReady = false;
          break;
        }
        requiredLevel = std::max(requiredLevel, predIt->second + 1);
      }
      if (!allPredecessorsReady)
        continue;

      auto lvlIt = levels.find(&op);
      if (lvlIt == levels.end()) {
        levels[&op] = requiredLevel;
        maxLevel = std::max(maxLevel, requiredLevel);
        changed = true;
      }
    }
  }

  return maxLevel;
}

static bool verifyDependencyLevels(
    Block *body, const DenseMap<Operation *, int> &levels) {
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    if (!levels.count(&op)) {
      llvm::errs() << "[cv-split] No dependency level assigned to " << op.getName()
                   << "; dependency graph may contain a cycle, bail\n";
      return false;
    }
  }
  return true;
}

// ============================================================================
// Stage 6: Level diagnostics
// ============================================================================
static void logLevelHistogram(
    Block *body,
    const DenseMap<Operation *, int> &levels,
    const DenseMap<Operation *, EngineType> &classification,
    int maxLevel) {
  for (int lvl = 0; lvl <= maxLevel; ++lvl) {
    SmallVector<Operation *> cubeOps, vectorOps;
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(&op))
        continue;
      auto lvlIt = levels.find(&op);
      if (lvlIt->second != lvl)
        continue;
      auto classIt = classification.find(&op);
      if (classIt->second == EngineType::CUBE)
        cubeOps.push_back(&op);
      else
        vectorOps.push_back(&op);
    }

    llvm::errs() << "[cv-split]   L" << lvl << ": " << cubeOps.size()
                 << "C " << vectorOps.size() << "V\n";
  }
}

// ============================================================================
// Stage 7: Reorder by dependency level
// ============================================================================
static void reorderByLevel(Block *body,
                           const DenseMap<Operation *, int> &levels) {
  SmallVector<Operation *> ops;
  for (Operation &op : *body) {
    if (!isa<scf::YieldOp>(&op))
      ops.push_back(&op);
  }

  llvm::stable_sort(ops, [&](Operation *a, Operation *b) {
    return levels.lookup(a) < levels.lookup(b);
  });

  Operation *yield = body->getTerminator();
  for (auto *op : ops)
    op->moveBefore(yield);
}

// ============================================================================
// Dependency-level scheduler
// ----------------------------------------------------------------------------
// Orchestrates stages 4-7 over an (already unrolled) loop body:
//   1. build a def->use dependency graph (SSA edges + memref.copy->to_tensor
//      memory edges),
//   2. assign every op a level = longest dependency depth from a root,
//   3. report the per-level CUBE/VECTOR distribution,
//   4. reorder the body by level, ready to be
//      split into a CUBE scope and a VECTOR scope.
// run() returns false when dependency levels cannot be assigned to every op.
// ============================================================================
bool DependencyScheduler::run(
    Block *body,
    const DenseMap<Operation *, EngineType> &classification) {
  buildDependencyGraph(body, predecessors, successors);
  if (!validateDependencyGraph(predecessors))
    return false;

  SmallVector<Operation *> roots = collectRoots(body, predecessors);
  llvm::errs() << "[cv-split] " << roots.size() << " roots\n";

  maxLevel = assignDependencyLevels(body, predecessors, roots, levels);
  if (!verifyDependencyLevels(body, levels))
    return false;
  llvm::errs() << "[cv-split] " << (maxLevel + 1) << " dependency levels\n";

  logLevelHistogram(body, levels, classification, maxLevel);

  reorderByLevel(body, levels);
  llvm::errs() << "[cv-split] Reordered by level\n";
  return true;
}

} // namespace mlir::triton::cv_split
