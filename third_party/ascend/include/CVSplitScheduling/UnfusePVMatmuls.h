#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_UNFUSE_PV_MATMULS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_UNFUSE_PV_MATMULS_H

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"

namespace mlir::triton::cv_split {

LogicalResult unfusePVMatmuls(Block *body, Classification &classification);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_UNFUSE_PV_MATMULS_H
