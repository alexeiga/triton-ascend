#ifndef TRITON_CV_SPLIT_SCHEDULING_PRE_CHECK_H
#define TRITON_CV_SPLIT_SCHEDULING_PRE_CHECK_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton {

/// Checks the structural assumptions that must hold before CV split scheduling
/// mutates `funcOp`. Returns the single supported candidate loop on success.
FailureOr<scf::ForOp> preCheckCVSplitScheduling(func::FuncOp funcOp,
                                               int unrollFactor);

} // namespace mlir::triton

#endif // TRITON_CV_SPLIT_SCHEDULING_PRE_CHECK_H
