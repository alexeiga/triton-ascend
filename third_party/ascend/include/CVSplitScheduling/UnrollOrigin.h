#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_UNROLL_ORIGIN_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_UNROLL_ORIGIN_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::cv_split {

inline constexpr llvm::StringLiteral kUnrollOriginIdAttrName =
    "cv_split.origin_id";

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_UNROLL_ORIGIN_H
