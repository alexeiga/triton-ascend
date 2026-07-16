#include "ascend/include/CVSplitScheduling/DependencyScheduler.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
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
  // to_tensor reads from the same alloc. Without this edge, BFS leveling
  // can place to_tensor BEFORE copy, causing reads of uninitialized data.
  for (Operation &op : *body) {
    auto copyOp = dyn_cast<memref::CopyOp>(&op);
    if (!copyOp) continue;
    Value dst = copyOp.getTarget();
    for (Operation *user : dst.getUsers()) {
      if (user == &op || user->getBlock() != body) continue;
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

// ============================================================================
// Stage 5: BFS levelization
// ============================================================================
static SmallVector<Operation *> collectRoots(
    Block *body,
    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors) {
  SmallVector<Operation *> roots;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    auto it = predecessors.find(&op);
    if (it == predecessors.end() || it->second.empty())
      roots.push_back(&op);
  }
  return roots;
}

static int bfsLevelize(
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
      if (it == predecessors.end()) {
        if (!levels.count(&op)) {
          levels[&op] = 0;
          changed = true;
        }
        continue;
      }

      int requiredLevel = 0;
      for (auto *pred : it->second) {
        auto predIt = levels.find(pred);
        if (predIt != levels.end())
          requiredLevel = std::max(requiredLevel, predIt->second + 1);
      }

      auto lvlIt = levels.find(&op);
      if (lvlIt == levels.end() || requiredLevel > lvlIt->second) {
        levels[&op] = requiredLevel;
        maxLevel = std::max(maxLevel, requiredLevel);
        changed = true;
      }
    }
  }

  return maxLevel;
}

// ============================================================================
// Stage 6: Level purity check
// ============================================================================
static bool checkLevelPurity(
    Block *body,
    const DenseMap<Operation *, int> &levels,
    const DenseMap<Operation *, EngineType> &classification,
    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
    int maxLevel) {
  for (int lvl = 0; lvl <= maxLevel; ++lvl) {
    SmallVector<Operation *> cubeOps, vectorOps;
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(&op)) continue;
      auto lvlIt = levels.find(&op);
      if (lvlIt == levels.end() || lvlIt->second != lvl) continue;
      auto classIt = classification.find(&op);
      if (classIt == classification.end()) continue;
      if (classIt->second == EngineType::CUBE)
        cubeOps.push_back(&op);
      else
        vectorOps.push_back(&op);
    }

    if (cubeOps.empty() || vectorOps.empty())
      continue;

    llvm::errs() << "[cv-split] Level " << lvl << " mixed ("
                 << cubeOps.size() << "C, " << vectorOps.size() << "V)\n";

    DenseSet<Operation *> cubeSet(cubeOps.begin(), cubeOps.end());
    DenseSet<Operation *> vectorSet(vectorOps.begin(), vectorOps.end());

    for (auto *cOp : cubeOps) {
      auto predIt = predecessors.find(cOp);
      if (predIt == predecessors.end()) continue;
      for (auto *pred : predIt->second) {
        if (vectorSet.count(pred)) {
          llvm::errs() << "[cv-split] Level " << lvl
                       << ": CUBE depends on VECTOR -- bail\n";
          return false;
        }
      }
    }
    for (auto *vOp : vectorOps) {
      auto predIt = predecessors.find(vOp);
      if (predIt == predecessors.end()) continue;
      for (auto *pred : predIt->second) {
        if (cubeSet.count(pred)) {
          llvm::errs() << "[cv-split] Level " << lvl
                       << ": VECTOR depends on CUBE -- bail\n";
          return false;
        }
      }
    }
  }
  return true;
}

// ============================================================================
// Stage 7: Reorder by BFS level
// ============================================================================
static void reorderByLevel(Block *body,
                           const DenseMap<Operation *, int> &levels) {
  SmallVector<Operation *> ops;
  for (Operation &op : *body) {
    if (!isa<scf::YieldOp>(&op))
      ops.push_back(&op);
  }

  llvm::stable_sort(ops, [&](Operation *a, Operation *b) {
    int la = 0, lb = 0;
    auto itA = levels.find(a);
    if (itA != levels.end()) la = itA->second;
    auto itB = levels.find(b);
    if (itB != levels.end()) lb = itB->second;
    return la < lb;
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
//   2. assign every op a BFS "level" = longest dependency depth from a root,
//   3. verify the levels are cleanly separable (no level contains a CUBE op and
//      a VECTOR op that depend on each other),
//   4. reorder the body by level so same-engine work is grouped, ready to be
//      split into a CUBE scope and a VECTOR scope.
// run() returns false (leaving the body untouched) when the work is entangled
// and cannot be cleanly separated, so the caller can bail safely.
// ============================================================================
bool DependencyScheduler::run(
    Block *body,
    const DenseMap<Operation *, EngineType> &classification) {
  buildDependencyGraph(body, predecessors, successors);

  SmallVector<Operation *> roots = collectRoots(body, predecessors);
  llvm::errs() << "[cv-split] " << roots.size() << " roots\n";

  maxLevel = bfsLevelize(body, predecessors, roots, levels);
  llvm::errs() << "[cv-split] " << (maxLevel + 1) << " BFS levels\n";
  logLevelHistogram(body, classification);

  if (!checkLevelPurity(body, levels, classification, predecessors,
                        maxLevel)) {
    llvm::errs() << "[cv-split] Purity check failed, bail\n";
    return false;
  }
  llvm::errs() << "[cv-split] Purity OK\n";

  reorderByLevel(body, levels);
  llvm::errs() << "[cv-split] Reordered by level\n";
  return true;
}

// Per-level CUBE/VECTOR op-count breakdown (diagnostic only).
void DependencyScheduler::logLevelHistogram(
    Block *body,
    const DenseMap<Operation *, EngineType> &classification) const {
  for (int lvl = 0; lvl <= maxLevel; ++lvl) {
    int nC = 0, nV = 0;
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(&op)) continue;
      auto lvlIt = levels.find(&op);
      if (lvlIt == levels.end() || lvlIt->second != lvl) continue;
      auto clsIt = classification.find(&op);
      if (clsIt == classification.end()) continue;
      if (clsIt->second == EngineType::CUBE) ++nC; else ++nV;
    }
    llvm::errs() << "[cv-split]   L" << lvl << ": "
                 << nC << "C " << nV << "V\n";
  }
}

} // namespace mlir::triton::cv_split
