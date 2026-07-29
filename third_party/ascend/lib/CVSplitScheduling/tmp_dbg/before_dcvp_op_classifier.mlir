module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c524288_i64 = arith.constant 524288 : i64
    %c0_i32 = arith.constant 0 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = tensor.empty() : tensor<32xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
    scf.for %arg13 = %arg10 to %c1_i32 step %c32_i32  : i32 {
      %8 = arith.divsi %arg13, %c256_i32 : i32
      %9 = arith.remsi %arg13, %c256_i32 : i32
      %10 = arith.extsi %8 : i32 to i64
      %11 = arith.muli %10, %c524288_i64 : i64
      %12 = arith.index_cast %11 : i64 to index
      %13 = arith.muli %9, %c32_i32 : i32
      %14 = arith.maxsi %13, %c0_i32 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %15, %c64 : index
      %17 = arith.addi %16, %12 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%17], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x64xf16>
      memref.copy %reinterpret_cast, %alloc : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
      %18 = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16>
      %c128_i32 = arith.constant 128 : i32
      %19:5 = scf.for %arg14 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg15 = %7, %arg16 = %1, %arg17 = %6, %arg18 = %c0_i32, %arg19 = %c0_i32) -> (tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32)  : i32 {
        %28 = arith.maxsi %arg18, %c0_i32 {cv_split.origin_id = 0 : i64} : i32
        %29 = arith.index_cast %28 {cv_split.origin_id = 1 : i64} : i32 to index
        %30 = arith.muli %29, %c64 {cv_split.origin_id = 2 : i64} : index
        %31 = arith.addi %30, %12 {cv_split.origin_id = 3 : i64} : index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 4 : i64} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %32 = arith.maxsi %arg19, %c0_i32 {cv_split.origin_id = 5 : i64} : i32
        %33 = arith.index_cast %32 {cv_split.origin_id = 6 : i64} : i32 to index
        %34 = arith.muli %33, %c64 {cv_split.origin_id = 7 : i64} : index
        %35 = arith.addi %34, %12 {cv_split.origin_id = 8 : i64} : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 9 : i64} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_7 = memref.alloc() {cv_split.origin_id = 10 : i64} : memref<32x64xf16>
        memref.copy %reinterpret_cast_6, %alloc_7 {cv_split.origin_id = 11 : i64} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %36 = bufferization.to_tensor %alloc_7 restrict writable {cv_split.origin_id = 12 : i64} : memref<32x64xf16>
        %37 = tensor.empty() {cv_split.origin_id = 13 : i64} : tensor<64x32xf16>
        %transposed = linalg.transpose ins(%36 : tensor<32x64xf16>) outs(%37 : tensor<64x32xf16>) permutation = [1, 0]  {cv_split.origin_id = 14 : i64}
        %38 = linalg.matmul {cv_split.origin_id = 15 : i64, input_precision = "ieee"} ins(%18, %transposed : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %39 = arith.mulf %38, %3 {cv_split.origin_id = 16 : i64} : tensor<32x32xf32>
        %reduced = linalg.reduce ins(%39 : tensor<32x32xf32>) outs(%6 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 17 : i64}
          (%in: f32, %init: f32) {
            %132 = arith.maximumf %in, %init : f32
            linalg.yield %132 : f32
          }
        %40 = arith.maximumf %arg17, %reduced {cv_split.origin_id = 18 : i64} : tensor<32xf32>
        %broadcasted_8 = linalg.broadcast ins(%40 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {cv_split.origin_id = 19 : i64}
        %41 = arith.subf %39, %broadcasted_8 {cv_split.origin_id = 20 : i64} : tensor<32x32xf32>
        %42 = math.exp %41 {cv_split.origin_id = 21 : i64} : tensor<32x32xf32>
        %43 = arith.truncf %42 {cv_split.origin_id = 22 : i64} : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_9 = memref.alloc() {cv_split.origin_id = 23 : i64} : memref<32x64xf16>
        memref.copy %reinterpret_cast_5, %alloc_9 {cv_split.origin_id = 24 : i64} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %44 = bufferization.to_tensor %alloc_9 restrict writable {cv_split.origin_id = 25 : i64} : memref<32x64xf16>
        %45 = linalg.fill {cv_split.origin_id = 26 : i64} ins(%cst_2 : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_10 = linalg.reduce ins(%42 : tensor<32x32xf32>) outs(%45 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 27 : i64}
          (%in: f32, %init: f32) {
            %132 = arith.addf %in, %init : f32
            linalg.yield %132 : f32
          }
        %46 = arith.subf %arg17, %40 {cv_split.origin_id = 28 : i64} : tensor<32xf32>
        %47 = math.exp %46 {cv_split.origin_id = 29 : i64} : tensor<32xf32>
        %48 = arith.mulf %arg15, %47 {cv_split.origin_id = 30 : i64} : tensor<32xf32>
        %49 = arith.addf %48, %reduced_10 {cv_split.origin_id = 31 : i64} : tensor<32xf32>
        %broadcasted_11 = linalg.broadcast ins(%47 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {cv_split.origin_id = 32 : i64}
        %50 = arith.mulf %arg16, %broadcasted_11 {cv_split.origin_id = 33 : i64} : tensor<32x64xf32>
        %51 = linalg.matmul {cv_split.origin_id = 34 : i64, input_precision = "ieee"} ins(%43, %44 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%50 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %52 = arith.addi %arg18, %c32_i32 {cv_split.origin_id = 35 : i64} : i32
        %53 = arith.addi %arg19, %c32_i32 {cv_split.origin_id = 36 : i64} : i32
        %54 = arith.maxsi %52, %c0_i32 {cv_split.origin_id = 0 : i64} : i32
        %55 = arith.index_cast %54 {cv_split.origin_id = 1 : i64} : i32 to index
        %56 = arith.muli %55, %c64 {cv_split.origin_id = 2 : i64} : index
        %57 = arith.addi %56, %12 {cv_split.origin_id = 3 : i64} : index
        %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%57], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 4 : i64} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %58 = arith.maxsi %53, %c0_i32 {cv_split.origin_id = 5 : i64} : i32
        %59 = arith.index_cast %58 {cv_split.origin_id = 6 : i64} : i32 to index
        %60 = arith.muli %59, %c64 {cv_split.origin_id = 7 : i64} : index
        %61 = arith.addi %60, %12 {cv_split.origin_id = 8 : i64} : index
        %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%61], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 9 : i64} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_14 = memref.alloc() {cv_split.origin_id = 10 : i64} : memref<32x64xf16>
        memref.copy %reinterpret_cast_13, %alloc_14 {cv_split.origin_id = 11 : i64} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %62 = bufferization.to_tensor %alloc_14 restrict writable {cv_split.origin_id = 12 : i64} : memref<32x64xf16>
        %63 = tensor.empty() {cv_split.origin_id = 13 : i64} : tensor<64x32xf16>
        %transposed_15 = linalg.transpose ins(%62 : tensor<32x64xf16>) outs(%63 : tensor<64x32xf16>) permutation = [1, 0]  {cv_split.origin_id = 14 : i64}
        %64 = linalg.matmul {cv_split.origin_id = 15 : i64, input_precision = "ieee"} ins(%18, %transposed_15 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %65 = arith.mulf %64, %3 {cv_split.origin_id = 16 : i64} : tensor<32x32xf32>
        %reduced_16 = linalg.reduce ins(%65 : tensor<32x32xf32>) outs(%6 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 17 : i64}
          (%in: f32, %init: f32) {
            %132 = arith.maximumf %in, %init : f32
            linalg.yield %132 : f32
          }
        %66 = arith.maximumf %40, %reduced_16 {cv_split.origin_id = 18 : i64} : tensor<32xf32>
        %broadcasted_17 = linalg.broadcast ins(%66 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {cv_split.origin_id = 19 : i64}
        %67 = arith.subf %65, %broadcasted_17 {cv_split.origin_id = 20 : i64} : tensor<32x32xf32>
        %68 = math.exp %67 {cv_split.origin_id = 21 : i64} : tensor<32x32xf32>
        %69 = arith.truncf %68 {cv_split.origin_id = 22 : i64} : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_18 = memref.alloc() {cv_split.origin_id = 23 : i64} : memref<32x64xf16>
        memref.copy %reinterpret_cast_12, %alloc_18 {cv_split.origin_id = 24 : i64} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %70 = bufferization.to_tensor %alloc_18 restrict writable {cv_split.origin_id = 25 : i64} : memref<32x64xf16>
        %71 = linalg.fill {cv_split.origin_id = 26 : i64} ins(%cst_2 : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_19 = linalg.reduce ins(%68 : tensor<32x32xf32>) outs(%71 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 27 : i64}
          (%in: f32, %init: f32) {
            %132 = arith.addf %in, %init : f32
            linalg.yield %132 : f32
          }
        %72 = arith.subf %40, %66 {cv_split.origin_id = 28 : i64} : tensor<32xf32>
        %73 = math.exp %72 {cv_split.origin_id = 29 : i64} : tensor<32xf32>
        %74 = arith.mulf %49, %73 {cv_split.origin_id = 30 : i64} : tensor<32xf32>
        %75 = arith.addf %74, %reduced_19 {cv_split.origin_id = 31 : i64} : tensor<32xf32>
        %broadcasted_20 = linalg.broadcast ins(%73 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {cv_split.origin_id = 32 : i64}
        %76 = arith.mulf %51, %broadcasted_20 {cv_split.origin_id = 33 : i64} : tensor<32x64xf32>
        %77 = linalg.matmul {cv_split.origin_id = 34 : i64, input_precision = "ieee"} ins(%69, %70 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%76 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %78 = arith.addi %52, %c32_i32 {cv_split.origin_id = 35 : i64} : i32
        %79 = arith.addi %53, %c32_i32 {cv_split.origin_id = 36 : i64} : i32
        %80 = arith.maxsi %78, %c0_i32 {cv_split.origin_id = 0 : i64} : i32
        %81 = arith.index_cast %80 {cv_split.origin_id = 1 : i64} : i32 to index
        %82 = arith.muli %81, %c64 {cv_split.origin_id = 2 : i64} : index
        %83 = arith.addi %82, %12 {cv_split.origin_id = 3 : i64} : index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg4 to offset: [%83], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 4 : i64} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %84 = arith.maxsi %79, %c0_i32 {cv_split.origin_id = 5 : i64} : i32
        %85 = arith.index_cast %84 {cv_split.origin_id = 6 : i64} : i32 to index
        %86 = arith.muli %85, %c64 {cv_split.origin_id = 7 : i64} : index
        %87 = arith.addi %86, %12 {cv_split.origin_id = 8 : i64} : index
        %reinterpret_cast_22 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 9 : i64} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_23 = memref.alloc() {cv_split.origin_id = 10 : i64} : memref<32x64xf16>
        memref.copy %reinterpret_cast_22, %alloc_23 {cv_split.origin_id = 11 : i64} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %88 = bufferization.to_tensor %alloc_23 restrict writable {cv_split.origin_id = 12 : i64} : memref<32x64xf16>
        %89 = tensor.empty() {cv_split.origin_id = 13 : i64} : tensor<64x32xf16>
        %transposed_24 = linalg.transpose ins(%88 : tensor<32x64xf16>) outs(%89 : tensor<64x32xf16>) permutation = [1, 0]  {cv_split.origin_id = 14 : i64}
        %90 = linalg.matmul {cv_split.origin_id = 15 : i64, input_precision = "ieee"} ins(%18, %transposed_24 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %91 = arith.mulf %90, %3 {cv_split.origin_id = 16 : i64} : tensor<32x32xf32>
        %reduced_25 = linalg.reduce ins(%91 : tensor<32x32xf32>) outs(%6 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 17 : i64}
          (%in: f32, %init: f32) {
            %132 = arith.maximumf %in, %init : f32
            linalg.yield %132 : f32
          }
        %92 = arith.maximumf %66, %reduced_25 {cv_split.origin_id = 18 : i64} : tensor<32xf32>
        %broadcasted_26 = linalg.broadcast ins(%92 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {cv_split.origin_id = 19 : i64}
        %93 = arith.subf %91, %broadcasted_26 {cv_split.origin_id = 20 : i64} : tensor<32x32xf32>
        %94 = math.exp %93 {cv_split.origin_id = 21 : i64} : tensor<32x32xf32>
        %95 = arith.truncf %94 {cv_split.origin_id = 22 : i64} : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_27 = memref.alloc() {cv_split.origin_id = 23 : i64} : memref<32x64xf16>
        memref.copy %reinterpret_cast_21, %alloc_27 {cv_split.origin_id = 24 : i64} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %96 = bufferization.to_tensor %alloc_27 restrict writable {cv_split.origin_id = 25 : i64} : memref<32x64xf16>
        %97 = linalg.fill {cv_split.origin_id = 26 : i64} ins(%cst_2 : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_28 = linalg.reduce ins(%94 : tensor<32x32xf32>) outs(%97 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 27 : i64}
          (%in: f32, %init: f32) {
            %132 = arith.addf %in, %init : f32
            linalg.yield %132 : f32
          }
        %98 = arith.subf %66, %92 {cv_split.origin_id = 28 : i64} : tensor<32xf32>
        %99 = math.exp %98 {cv_split.origin_id = 29 : i64} : tensor<32xf32>
        %100 = arith.mulf %75, %99 {cv_split.origin_id = 30 : i64} : tensor<32xf32>
        %101 = arith.addf %100, %reduced_28 {cv_split.origin_id = 31 : i64} : tensor<32xf32>
        %broadcasted_29 = linalg.broadcast ins(%99 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {cv_split.origin_id = 32 : i64}
        %102 = arith.mulf %77, %broadcasted_29 {cv_split.origin_id = 33 : i64} : tensor<32x64xf32>
        %103 = linalg.matmul {cv_split.origin_id = 34 : i64, input_precision = "ieee"} ins(%95, %96 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%102 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %104 = arith.addi %78, %c32_i32 {cv_split.origin_id = 35 : i64} : i32
        %105 = arith.addi %79, %c32_i32 {cv_split.origin_id = 36 : i64} : i32
        %106 = arith.maxsi %104, %c0_i32 {cv_split.origin_id = 0 : i64} : i32
        %107 = arith.index_cast %106 {cv_split.origin_id = 1 : i64} : i32 to index
        %108 = arith.muli %107, %c64 {cv_split.origin_id = 2 : i64} : index
        %109 = arith.addi %108, %12 {cv_split.origin_id = 3 : i64} : index
        %reinterpret_cast_30 = memref.reinterpret_cast %arg4 to offset: [%109], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 4 : i64} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %110 = arith.maxsi %105, %c0_i32 {cv_split.origin_id = 5 : i64} : i32
        %111 = arith.index_cast %110 {cv_split.origin_id = 6 : i64} : i32 to index
        %112 = arith.muli %111, %c64 {cv_split.origin_id = 7 : i64} : index
        %113 = arith.addi %112, %12 {cv_split.origin_id = 8 : i64} : index
        %reinterpret_cast_31 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 9 : i64} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_32 = memref.alloc() {cv_split.origin_id = 10 : i64} : memref<32x64xf16>
        memref.copy %reinterpret_cast_31, %alloc_32 {cv_split.origin_id = 11 : i64} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %114 = bufferization.to_tensor %alloc_32 restrict writable {cv_split.origin_id = 12 : i64} : memref<32x64xf16>
        %115 = tensor.empty() {cv_split.origin_id = 13 : i64} : tensor<64x32xf16>
        %transposed_33 = linalg.transpose ins(%114 : tensor<32x64xf16>) outs(%115 : tensor<64x32xf16>) permutation = [1, 0]  {cv_split.origin_id = 14 : i64}
        %116 = linalg.matmul {cv_split.origin_id = 15 : i64, input_precision = "ieee"} ins(%18, %transposed_33 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %117 = arith.mulf %116, %3 {cv_split.origin_id = 16 : i64} : tensor<32x32xf32>
        %reduced_34 = linalg.reduce ins(%117 : tensor<32x32xf32>) outs(%6 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 17 : i64}
          (%in: f32, %init: f32) {
            %132 = arith.maximumf %in, %init : f32
            linalg.yield %132 : f32
          }
        %118 = arith.maximumf %92, %reduced_34 {cv_split.origin_id = 18 : i64} : tensor<32xf32>
        %broadcasted_35 = linalg.broadcast ins(%118 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {cv_split.origin_id = 19 : i64}
        %119 = arith.subf %117, %broadcasted_35 {cv_split.origin_id = 20 : i64} : tensor<32x32xf32>
        %120 = math.exp %119 {cv_split.origin_id = 21 : i64} : tensor<32x32xf32>
        %121 = arith.truncf %120 {cv_split.origin_id = 22 : i64} : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_36 = memref.alloc() {cv_split.origin_id = 23 : i64} : memref<32x64xf16>
        memref.copy %reinterpret_cast_30, %alloc_36 {cv_split.origin_id = 24 : i64} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %122 = bufferization.to_tensor %alloc_36 restrict writable {cv_split.origin_id = 25 : i64} : memref<32x64xf16>
        %123 = linalg.fill {cv_split.origin_id = 26 : i64} ins(%cst_2 : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_37 = linalg.reduce ins(%120 : tensor<32x32xf32>) outs(%123 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 27 : i64}
          (%in: f32, %init: f32) {
            %132 = arith.addf %in, %init : f32
            linalg.yield %132 : f32
          }
        %124 = arith.subf %92, %118 {cv_split.origin_id = 28 : i64} : tensor<32xf32>
        %125 = math.exp %124 {cv_split.origin_id = 29 : i64} : tensor<32xf32>
        %126 = arith.mulf %101, %125 {cv_split.origin_id = 30 : i64} : tensor<32xf32>
        %127 = arith.addf %126, %reduced_37 {cv_split.origin_id = 31 : i64} : tensor<32xf32>
        %broadcasted_38 = linalg.broadcast ins(%125 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {cv_split.origin_id = 32 : i64}
        %128 = arith.mulf %103, %broadcasted_38 {cv_split.origin_id = 33 : i64} : tensor<32x64xf32>
        %129 = linalg.matmul {cv_split.origin_id = 34 : i64, input_precision = "ieee"} ins(%121, %122 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%128 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %130 = arith.addi %104, %c32_i32 {cv_split.origin_id = 35 : i64} : i32
        %131 = arith.addi %105, %c32_i32 {cv_split.origin_id = 36 : i64} : i32
        scf.yield %127, %129, %118, %130, %131 : tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32
      } {tt.divisibility_arg1 = dense<32> : tensor<1xi32>}
      %broadcasted = linalg.broadcast ins(%19#0 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1] 
      %20 = arith.divf %19#1, %broadcasted : tensor<32x64xf32>
      %21 = math.log %19#0 : tensor<32xf32>
      %22 = arith.addf %19#2, %21 : tensor<32xf32>
      %23 = arith.muli %8, %c8192_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.index_cast %13 : i32 to index
      %26 = arith.addi %24, %25 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %22 in writable %reinterpret_cast_4 : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
      %27 = arith.truncf %20 : tensor<32x64xf32> to tensor<32x64xf16>
      bufferization.materialize_in_destination %27 in writable %reinterpret_cast_3 : (tensor<32x64xf16>, memref<32x64xf16, strided<[64, 1], offset: ?>>) -> ()
    }
    return
  }
}

