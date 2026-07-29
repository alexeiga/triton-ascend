module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant {ssbuffer.core_type = "VECTOR"} 64 : index
    %c64_0 = arith.constant {ssbuffer.core_type = "CUBE"} 64 : index
    %cst = arith.constant {ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %cst_1 = arith.constant {ssbuffer.core_type = "VECTOR"} 0xFF800000 : f32
    %cst_2 = arith.constant {ssbuffer.core_type = "VECTOR"} 1.250000e-01 : f32
    %c32_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 32 : i32
    %c32_i32_3 = arith.constant {ssbuffer.core_type = "CUBE"} 32 : i32
    %c256_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 256 : i32
    %c256_i32_4 = arith.constant {ssbuffer.core_type = "CUBE"} 256 : i32
    %c524288_i64 = arith.constant {ssbuffer.core_type = "VECTOR"} 524288 : i64
    %c524288_i64_5 = arith.constant {ssbuffer.core_type = "CUBE"} 524288 : i64
    %c0_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : i32
    %c0_i32_6 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : i32
    %c8192_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 8192 : i32
    %c1_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %cst_7 = arith.constant {ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %cst_8 = arith.constant {ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %0 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<32x64xf32>
    %1 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_7 : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %2 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %3 = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %4 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_2 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst_8 : f32) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
    %7 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_1 : f32) outs(%6 : tensor<32xf32>) -> tensor<32xf32>
    %8 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%6 : tensor<32xf32>) -> tensor<32xf32>
    scf.for %arg13 = %arg10 to %c1_i32 step %c32_i32_3  : i32 {
      %9 = arith.divsi %arg13, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %10 = arith.divsi %arg13, %c256_i32_4 {ssbuffer.core_type = "CUBE"} : i32
      %11 = arith.remsi %arg13, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %12 = arith.remsi %arg13, %c256_i32_4 {ssbuffer.core_type = "CUBE"} : i32
      %13 = arith.extsi %9 {ssbuffer.core_type = "VECTOR"} : i32 to i64
      %14 = arith.extsi %10 {ssbuffer.core_type = "CUBE"} : i32 to i64
      %15 = arith.muli %13, %c524288_i64 {ssbuffer.core_type = "VECTOR"} : i64
      %16 = arith.muli %14, %c524288_i64_5 {ssbuffer.core_type = "CUBE"} : i64
      %17 = arith.index_cast %15 {ssbuffer.core_type = "VECTOR"} : i64 to index
      %18 = arith.index_cast %16 {ssbuffer.core_type = "CUBE"} : i64 to index
      %19 = arith.muli %11, %c32_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %20 = arith.muli %12, %c32_i32_3 {ssbuffer.core_type = "CUBE"} : i32
      %21 = arith.maxsi %19, %c0_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %22 = arith.maxsi %20, %c0_i32_6 {ssbuffer.core_type = "CUBE"} : i32
      %23 = arith.index_cast %21 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %24 = arith.index_cast %22 {ssbuffer.core_type = "CUBE"} : i32 to index
      %25 = arith.muli %23, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %26 = arith.muli %24, %c64_0 {ssbuffer.core_type = "CUBE"} : index
      %27 = arith.addi %25, %17 {ssbuffer.core_type = "VECTOR"} : index
      %28 = arith.addi %26, %18 {ssbuffer.core_type = "CUBE"} : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [32, 64], strides: [64, 1] {ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%27], sizes: [32, 64], strides: [64, 1] {ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
      memref.copy %reinterpret_cast, %alloc {ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
      %29 = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
      %c128_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 128 : i32
      %30:5 = scf.for %arg14 = %c0_i32_6 to %c8192_i32 step %c128_i32 iter_args(%arg15 = %8, %arg16 = %1, %arg17 = %7, %arg18 = %c0_i32_6, %arg19 = %c0_i32_6) -> (tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32)  : i32 {
        %39 = arith.maxsi %arg18, %c0_i32_6 {cv_split.origin_id = 0 : i64, ssbuffer.core_type = "CUBE"} : i32
        %40 = arith.index_cast %39 {cv_split.origin_id = 1 : i64, ssbuffer.core_type = "CUBE"} : i32 to index
        %41 = arith.muli %40, %c64_0 {cv_split.origin_id = 2 : i64, ssbuffer.core_type = "CUBE"} : index
        %42 = arith.addi %41, %18 {cv_split.origin_id = 3 : i64, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 4 : i64, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %43 = arith.maxsi %arg19, %c0_i32_6 {cv_split.origin_id = 5 : i64, ssbuffer.core_type = "CUBE"} : i32
        %44 = arith.index_cast %43 {cv_split.origin_id = 6 : i64, ssbuffer.core_type = "CUBE"} : i32 to index
        %45 = arith.muli %44, %c64_0 {cv_split.origin_id = 7 : i64, ssbuffer.core_type = "CUBE"} : index
        %46 = arith.addi %45, %18 {cv_split.origin_id = 8 : i64, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%46], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 9 : i64, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_13 = memref.alloc() {cv_split.origin_id = 10 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        memref.copy %reinterpret_cast_12, %alloc_13 {cv_split.origin_id = 11 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %47 = bufferization.to_tensor %alloc_13 restrict writable {cv_split.origin_id = 12 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        %48 = tensor.empty() {cv_split.origin_id = 13 : i64, ssbuffer.core_type = "CUBE"} : tensor<64x32xf16>
        %transposed = linalg.transpose ins(%47 : tensor<32x64xf16>) outs(%48 : tensor<64x32xf16>) permutation = [1, 0]  {cv_split.origin_id = 14 : i64, ssbuffer.core_type = "CUBE"}
        %49 = linalg.matmul {cv_split.origin_id = 15 : i64, input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%29, %transposed : tensor<32x64xf16>, tensor<64x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %50 = arith.mulf %49, %4 {cv_split.origin_id = 16 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %reduced = linalg.reduce ins(%50 : tensor<32x32xf32>) outs(%7 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 17 : i64, ssbuffer.core_type = "VECTOR"}
          (%in: f32, %init: f32) {
            %143 = arith.maximumf %in, %init : f32
            linalg.yield %143 : f32
          }
        %51 = arith.maximumf %arg17, %reduced {cv_split.origin_id = 18 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %broadcasted_14 = linalg.broadcast ins(%51 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {cv_split.origin_id = 19 : i64, ssbuffer.core_type = "VECTOR"}
        %52 = arith.subf %50, %broadcasted_14 {cv_split.origin_id = 20 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %53 = math.exp %52 {cv_split.origin_id = 21 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %54 = arith.truncf %53 {cv_split.origin_id = 22 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_15 = memref.alloc() {cv_split.origin_id = 23 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        memref.copy %reinterpret_cast_11, %alloc_15 {cv_split.origin_id = 24 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %55 = bufferization.to_tensor %alloc_15 restrict writable {cv_split.origin_id = 25 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        %56 = linalg.fill {cv_split.origin_id = 26 : i64, ssbuffer.core_type = "VECTOR"} ins(%cst_7 : f32) outs(%6 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_16 = linalg.reduce ins(%53 : tensor<32x32xf32>) outs(%56 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 27 : i64, ssbuffer.core_type = "VECTOR"}
          (%in: f32, %init: f32) {
            %143 = arith.addf %in, %init : f32
            linalg.yield %143 : f32
          }
        %57 = arith.subf %arg17, %51 {cv_split.origin_id = 28 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %58 = math.exp %57 {cv_split.origin_id = 29 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %59 = arith.mulf %arg15, %58 {cv_split.origin_id = 30 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %60 = arith.addf %59, %reduced_16 {cv_split.origin_id = 31 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %broadcasted_17 = linalg.broadcast ins(%58 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {cv_split.origin_id = 32 : i64, ssbuffer.core_type = "VECTOR"}
        %61 = arith.mulf %arg16, %broadcasted_17 {cv_split.origin_id = 33 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x64xf32>
        %62 = linalg.matmul {cv_split.origin_id = 34 : i64, input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%54, %55 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%61 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %63 = arith.addi %arg18, %c32_i32_3 {cv_split.origin_id = 35 : i64, ssbuffer.core_type = "CUBE"} : i32
        %64 = arith.addi %arg19, %c32_i32_3 {cv_split.origin_id = 36 : i64, ssbuffer.core_type = "CUBE"} : i32
        %65 = arith.maxsi %63, %c0_i32_6 {cv_split.origin_id = 0 : i64, ssbuffer.core_type = "CUBE"} : i32
        %66 = arith.index_cast %65 {cv_split.origin_id = 1 : i64, ssbuffer.core_type = "CUBE"} : i32 to index
        %67 = arith.muli %66, %c64_0 {cv_split.origin_id = 2 : i64, ssbuffer.core_type = "CUBE"} : index
        %68 = arith.addi %67, %18 {cv_split.origin_id = 3 : i64, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_18 = memref.reinterpret_cast %arg4 to offset: [%68], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 4 : i64, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %69 = arith.maxsi %64, %c0_i32_6 {cv_split.origin_id = 5 : i64, ssbuffer.core_type = "CUBE"} : i32
        %70 = arith.index_cast %69 {cv_split.origin_id = 6 : i64, ssbuffer.core_type = "CUBE"} : i32 to index
        %71 = arith.muli %70, %c64_0 {cv_split.origin_id = 7 : i64, ssbuffer.core_type = "CUBE"} : index
        %72 = arith.addi %71, %18 {cv_split.origin_id = 8 : i64, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_19 = memref.reinterpret_cast %arg3 to offset: [%72], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 9 : i64, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_20 = memref.alloc() {cv_split.origin_id = 10 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        memref.copy %reinterpret_cast_19, %alloc_20 {cv_split.origin_id = 11 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %73 = bufferization.to_tensor %alloc_20 restrict writable {cv_split.origin_id = 12 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        %74 = tensor.empty() {cv_split.origin_id = 13 : i64, ssbuffer.core_type = "CUBE"} : tensor<64x32xf16>
        %transposed_21 = linalg.transpose ins(%73 : tensor<32x64xf16>) outs(%74 : tensor<64x32xf16>) permutation = [1, 0]  {cv_split.origin_id = 14 : i64, ssbuffer.core_type = "CUBE"}
        %75 = linalg.matmul {cv_split.origin_id = 15 : i64, input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%29, %transposed_21 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %76 = arith.mulf %75, %4 {cv_split.origin_id = 16 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %reduced_22 = linalg.reduce ins(%76 : tensor<32x32xf32>) outs(%7 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 17 : i64, ssbuffer.core_type = "VECTOR"}
          (%in: f32, %init: f32) {
            %143 = arith.maximumf %in, %init : f32
            linalg.yield %143 : f32
          }
        %77 = arith.maximumf %51, %reduced_22 {cv_split.origin_id = 18 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %broadcasted_23 = linalg.broadcast ins(%77 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {cv_split.origin_id = 19 : i64, ssbuffer.core_type = "VECTOR"}
        %78 = arith.subf %76, %broadcasted_23 {cv_split.origin_id = 20 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %79 = math.exp %78 {cv_split.origin_id = 21 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %80 = arith.truncf %79 {cv_split.origin_id = 22 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_24 = memref.alloc() {cv_split.origin_id = 23 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        memref.copy %reinterpret_cast_18, %alloc_24 {cv_split.origin_id = 24 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %81 = bufferization.to_tensor %alloc_24 restrict writable {cv_split.origin_id = 25 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        %82 = linalg.fill {cv_split.origin_id = 26 : i64, ssbuffer.core_type = "VECTOR"} ins(%cst_7 : f32) outs(%6 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_25 = linalg.reduce ins(%79 : tensor<32x32xf32>) outs(%82 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 27 : i64, ssbuffer.core_type = "VECTOR"}
          (%in: f32, %init: f32) {
            %143 = arith.addf %in, %init : f32
            linalg.yield %143 : f32
          }
        %83 = arith.subf %51, %77 {cv_split.origin_id = 28 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %84 = math.exp %83 {cv_split.origin_id = 29 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %85 = arith.mulf %60, %84 {cv_split.origin_id = 30 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %86 = arith.addf %85, %reduced_25 {cv_split.origin_id = 31 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %broadcasted_26 = linalg.broadcast ins(%84 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {cv_split.origin_id = 32 : i64, ssbuffer.core_type = "VECTOR"}
        %87 = arith.mulf %62, %broadcasted_26 {cv_split.origin_id = 33 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x64xf32>
        %88 = linalg.matmul {cv_split.origin_id = 34 : i64, input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%80, %81 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%87 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %89 = arith.addi %63, %c32_i32_3 {cv_split.origin_id = 35 : i64, ssbuffer.core_type = "CUBE"} : i32
        %90 = arith.addi %64, %c32_i32_3 {cv_split.origin_id = 36 : i64, ssbuffer.core_type = "CUBE"} : i32
        %91 = arith.maxsi %89, %c0_i32_6 {cv_split.origin_id = 0 : i64, ssbuffer.core_type = "CUBE"} : i32
        %92 = arith.index_cast %91 {cv_split.origin_id = 1 : i64, ssbuffer.core_type = "CUBE"} : i32 to index
        %93 = arith.muli %92, %c64_0 {cv_split.origin_id = 2 : i64, ssbuffer.core_type = "CUBE"} : index
        %94 = arith.addi %93, %18 {cv_split.origin_id = 3 : i64, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_27 = memref.reinterpret_cast %arg4 to offset: [%94], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 4 : i64, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %95 = arith.maxsi %90, %c0_i32_6 {cv_split.origin_id = 5 : i64, ssbuffer.core_type = "CUBE"} : i32
        %96 = arith.index_cast %95 {cv_split.origin_id = 6 : i64, ssbuffer.core_type = "CUBE"} : i32 to index
        %97 = arith.muli %96, %c64_0 {cv_split.origin_id = 7 : i64, ssbuffer.core_type = "CUBE"} : index
        %98 = arith.addi %97, %18 {cv_split.origin_id = 8 : i64, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_28 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 9 : i64, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_29 = memref.alloc() {cv_split.origin_id = 10 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        memref.copy %reinterpret_cast_28, %alloc_29 {cv_split.origin_id = 11 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %99 = bufferization.to_tensor %alloc_29 restrict writable {cv_split.origin_id = 12 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        %100 = tensor.empty() {cv_split.origin_id = 13 : i64, ssbuffer.core_type = "CUBE"} : tensor<64x32xf16>
        %transposed_30 = linalg.transpose ins(%99 : tensor<32x64xf16>) outs(%100 : tensor<64x32xf16>) permutation = [1, 0]  {cv_split.origin_id = 14 : i64, ssbuffer.core_type = "CUBE"}
        %101 = linalg.matmul {cv_split.origin_id = 15 : i64, input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%29, %transposed_30 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %102 = arith.mulf %101, %4 {cv_split.origin_id = 16 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %reduced_31 = linalg.reduce ins(%102 : tensor<32x32xf32>) outs(%7 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 17 : i64, ssbuffer.core_type = "VECTOR"}
          (%in: f32, %init: f32) {
            %143 = arith.maximumf %in, %init : f32
            linalg.yield %143 : f32
          }
        %103 = arith.maximumf %77, %reduced_31 {cv_split.origin_id = 18 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %broadcasted_32 = linalg.broadcast ins(%103 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {cv_split.origin_id = 19 : i64, ssbuffer.core_type = "VECTOR"}
        %104 = arith.subf %102, %broadcasted_32 {cv_split.origin_id = 20 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %105 = math.exp %104 {cv_split.origin_id = 21 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %106 = arith.truncf %105 {cv_split.origin_id = 22 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_33 = memref.alloc() {cv_split.origin_id = 23 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        memref.copy %reinterpret_cast_27, %alloc_33 {cv_split.origin_id = 24 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %107 = bufferization.to_tensor %alloc_33 restrict writable {cv_split.origin_id = 25 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        %108 = linalg.fill {cv_split.origin_id = 26 : i64, ssbuffer.core_type = "VECTOR"} ins(%cst_7 : f32) outs(%6 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_34 = linalg.reduce ins(%105 : tensor<32x32xf32>) outs(%108 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 27 : i64, ssbuffer.core_type = "VECTOR"}
          (%in: f32, %init: f32) {
            %143 = arith.addf %in, %init : f32
            linalg.yield %143 : f32
          }
        %109 = arith.subf %77, %103 {cv_split.origin_id = 28 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %110 = math.exp %109 {cv_split.origin_id = 29 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %111 = arith.mulf %86, %110 {cv_split.origin_id = 30 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %112 = arith.addf %111, %reduced_34 {cv_split.origin_id = 31 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %broadcasted_35 = linalg.broadcast ins(%110 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {cv_split.origin_id = 32 : i64, ssbuffer.core_type = "VECTOR"}
        %113 = arith.mulf %88, %broadcasted_35 {cv_split.origin_id = 33 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x64xf32>
        %114 = linalg.matmul {cv_split.origin_id = 34 : i64, input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%106, %107 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%113 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %115 = arith.addi %89, %c32_i32_3 {cv_split.origin_id = 35 : i64, ssbuffer.core_type = "CUBE"} : i32
        %116 = arith.addi %90, %c32_i32_3 {cv_split.origin_id = 36 : i64, ssbuffer.core_type = "CUBE"} : i32
        %117 = arith.maxsi %115, %c0_i32_6 {cv_split.origin_id = 0 : i64, ssbuffer.core_type = "CUBE"} : i32
        %118 = arith.index_cast %117 {cv_split.origin_id = 1 : i64, ssbuffer.core_type = "CUBE"} : i32 to index
        %119 = arith.muli %118, %c64_0 {cv_split.origin_id = 2 : i64, ssbuffer.core_type = "CUBE"} : index
        %120 = arith.addi %119, %18 {cv_split.origin_id = 3 : i64, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_36 = memref.reinterpret_cast %arg4 to offset: [%120], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 4 : i64, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %121 = arith.maxsi %116, %c0_i32_6 {cv_split.origin_id = 5 : i64, ssbuffer.core_type = "CUBE"} : i32
        %122 = arith.index_cast %121 {cv_split.origin_id = 6 : i64, ssbuffer.core_type = "CUBE"} : i32 to index
        %123 = arith.muli %122, %c64_0 {cv_split.origin_id = 7 : i64, ssbuffer.core_type = "CUBE"} : index
        %124 = arith.addi %123, %18 {cv_split.origin_id = 8 : i64, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_37 = memref.reinterpret_cast %arg3 to offset: [%124], sizes: [32, 64], strides: [64, 1] {cv_split.origin_id = 9 : i64, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_38 = memref.alloc() {cv_split.origin_id = 10 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        memref.copy %reinterpret_cast_37, %alloc_38 {cv_split.origin_id = 11 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %125 = bufferization.to_tensor %alloc_38 restrict writable {cv_split.origin_id = 12 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        %126 = tensor.empty() {cv_split.origin_id = 13 : i64, ssbuffer.core_type = "CUBE"} : tensor<64x32xf16>
        %transposed_39 = linalg.transpose ins(%125 : tensor<32x64xf16>) outs(%126 : tensor<64x32xf16>) permutation = [1, 0]  {cv_split.origin_id = 14 : i64, ssbuffer.core_type = "CUBE"}
        %127 = linalg.matmul {cv_split.origin_id = 15 : i64, input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%29, %transposed_39 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %128 = arith.mulf %127, %4 {cv_split.origin_id = 16 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %reduced_40 = linalg.reduce ins(%128 : tensor<32x32xf32>) outs(%7 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 17 : i64, ssbuffer.core_type = "VECTOR"}
          (%in: f32, %init: f32) {
            %143 = arith.maximumf %in, %init : f32
            linalg.yield %143 : f32
          }
        %129 = arith.maximumf %103, %reduced_40 {cv_split.origin_id = 18 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %broadcasted_41 = linalg.broadcast ins(%129 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]  {cv_split.origin_id = 19 : i64, ssbuffer.core_type = "VECTOR"}
        %130 = arith.subf %128, %broadcasted_41 {cv_split.origin_id = 20 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %131 = math.exp %130 {cv_split.origin_id = 21 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %132 = arith.truncf %131 {cv_split.origin_id = 22 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_42 = memref.alloc() {cv_split.origin_id = 23 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        memref.copy %reinterpret_cast_36, %alloc_42 {cv_split.origin_id = 24 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %133 = bufferization.to_tensor %alloc_42 restrict writable {cv_split.origin_id = 25 : i64, ssbuffer.core_type = "CUBE"} : memref<32x64xf16>
        %134 = linalg.fill {cv_split.origin_id = 26 : i64, ssbuffer.core_type = "VECTOR"} ins(%cst_7 : f32) outs(%6 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_43 = linalg.reduce ins(%131 : tensor<32x32xf32>) outs(%134 : tensor<32xf32>) dimensions = [1]  {cv_split.origin_id = 27 : i64, ssbuffer.core_type = "VECTOR"}
          (%in: f32, %init: f32) {
            %143 = arith.addf %in, %init : f32
            linalg.yield %143 : f32
          }
        %135 = arith.subf %103, %129 {cv_split.origin_id = 28 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %136 = math.exp %135 {cv_split.origin_id = 29 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %137 = arith.mulf %112, %136 {cv_split.origin_id = 30 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %138 = arith.addf %137, %reduced_43 {cv_split.origin_id = 31 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %broadcasted_44 = linalg.broadcast ins(%136 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {cv_split.origin_id = 32 : i64, ssbuffer.core_type = "VECTOR"}
        %139 = arith.mulf %114, %broadcasted_44 {cv_split.origin_id = 33 : i64, ssbuffer.core_type = "VECTOR"} : tensor<32x64xf32>
        %140 = linalg.matmul {cv_split.origin_id = 34 : i64, input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%132, %133 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%139 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %141 = arith.addi %115, %c32_i32_3 {cv_split.origin_id = 35 : i64, ssbuffer.core_type = "CUBE"} : i32
        %142 = arith.addi %116, %c32_i32_3 {cv_split.origin_id = 36 : i64, ssbuffer.core_type = "CUBE"} : i32
        scf.yield {ssbuffer.core_type = "VECTOR, CUBE, VECTOR, CUBE, CUBE"} %138, %140, %129, %141, %142 : tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32
      } {ssbuffer.core_type = "VECTOR, CUBE, VECTOR, CUBE, CUBE", tt.divisibility_arg1 = dense<32> : tensor<1xi32>}
      %broadcasted = linalg.broadcast ins(%30#0 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]  {ssbuffer.core_type = "VECTOR"}
      %31 = arith.divf %30#1, %broadcasted {ssbuffer.core_type = "VECTOR"} : tensor<32x64xf32>
      %32 = math.log %30#0 {ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
      %33 = arith.addf %30#2, %32 {ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
      %34 = arith.muli %9, %c8192_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %35 = arith.index_cast %34 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %36 = arith.index_cast %19 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %37 = arith.addi %35, %36 {ssbuffer.core_type = "VECTOR"} : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [32], strides: [1] {ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %33 in writable %reinterpret_cast_10 {ssbuffer.core_type = "VECTOR"} : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
      %38 = arith.truncf %31 {ssbuffer.core_type = "VECTOR"} : tensor<32x64xf32> to tensor<32x64xf16>
      bufferization.materialize_in_destination %38 in writable %reinterpret_cast_9 {ssbuffer.core_type = "VECTOR"} : (tensor<32x64xf16>, memref<32x64xf16, strided<[64, 1], offset: ?>>) -> ()
    }
    return
  }
}

