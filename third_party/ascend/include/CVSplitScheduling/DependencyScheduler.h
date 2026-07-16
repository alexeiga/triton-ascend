#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_DEPENDENCY_SCHEDULER_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_DEPENDENCY_SCHEDULER_H

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::cv_split {

class DependencyScheduler {
public:
  bool run(Block *body,
           const llvm::DenseMap<Operation *, EngineType> &classification);

private:
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> predecessors;
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> successors;
  llvm::DenseMap<Operation *, int> levels;
  int maxLevel = 0;

  void logLevelHistogram(
      Block *body,
      const llvm::DenseMap<Operation *, EngineType> &classification) const;
};

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_DEPENDENCY_SCHEDULER_H
