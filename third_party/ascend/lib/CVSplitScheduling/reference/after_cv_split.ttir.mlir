module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
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
      %c128_i32 = arith.constant 128 : i32
      %alloc = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"]} : memref<16x32xf32, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_3 {effects = ["write", "read"]} : memref<16x32xf32, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"]} : memref<16x64xf32, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"]} : memref<16x64xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<32x64xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_6 {mem_unique} : memref<32x64xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_6 {effects = ["write", "read"]} : memref<32x64xf16, #hivm.address_space<cbuf>>
      %alloc_7 = memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_7 {effects = ["write", "read"]} : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_8 = memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_8 {effects = ["write", "read"]} : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
      scope.scope : () -> () {
        %alloc_9 = memref.alloc() : memref<32x64xf16>
        memref.copy %reinterpret_cast, %alloc_9 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %29 = bufferization.to_tensor %alloc_9 restrict writable : memref<32x64xf16>
        annotation.mark %29 keys = ["bind_buffer"] values = [%alloc_6 : memref<32x64xf16, #hivm.address_space<cbuf>>] : tensor<32x64xf16>
        %memspacecast = memref.memory_space_cast %alloc_6 : memref<32x64xf16, #hivm.address_space<cbuf>> to memref<32x64xf16>
        %30 = bufferization.to_tensor %memspacecast restrict writable : memref<32x64xf16>
        %31 = hivm.hir.convert_layout %alloc_8 output_shape [32, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<2x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x32xf16, #hivm.address_space<cbuf>>
        %32 = hivm.hir.convert_layout %alloc_7 output_shape [32, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<2x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x32xf16, #hivm.address_space<cbuf>>
        %memspacecast_10 = memref.memory_space_cast %31 : memref<32x32xf16, #hivm.address_space<cbuf>> to memref<32x32xf16>
        %memspacecast_11 = memref.memory_space_cast %32 : memref<32x32xf16, #hivm.address_space<cbuf>> to memref<32x32xf16>
        %33:5 = scf.for %arg14 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg15 = %7, %arg16 = %1, %arg17 = %6, %arg18 = %c0_i32, %arg19 = %c0_i32) -> (tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32)  : i32 {
          %34 = arith.maxsi %arg18, %c0_i32 : i32
          %35 = arith.maxsi %arg19, %c0_i32 : i32
          %cst_12 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
          %36 = arith.addi %arg18, %c32_i32 : i32
          %37 = arith.addi %arg19, %c32_i32 : i32
          %cst_13 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
          %cst_14 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
          %cst_15 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
          %38 = arith.index_cast %34 : i32 to index
          %39 = arith.index_cast %35 : i32 to index
          %40 = arith.maxsi %36, %c0_i32 : i32
          %41 = arith.maxsi %37, %c0_i32 : i32
          %42 = arith.addi %36, %c32_i32 : i32
          %43 = arith.addi %37, %c32_i32 : i32
          %44 = arith.muli %38, %c64 : index
          %45 = arith.muli %39, %c64 : index
          %46 = arith.index_cast %40 : i32 to index
          %47 = arith.index_cast %41 : i32 to index
          %48 = arith.maxsi %42, %c0_i32 : i32
          %49 = arith.maxsi %43, %c0_i32 : i32
          %50 = arith.addi %42, %c32_i32 : i32
          %51 = arith.addi %43, %c32_i32 : i32
          %52 = arith.addi %44, %12 : index
          %53 = arith.addi %45, %12 : index
          %54 = arith.muli %46, %c64 : index
          %55 = arith.muli %47, %c64 : index
          %56 = arith.index_cast %48 : i32 to index
          %57 = arith.index_cast %49 : i32 to index
          %58 = arith.maxsi %50, %c0_i32 : i32
          %59 = arith.maxsi %51, %c0_i32 : i32
          %60 = arith.addi %50, %c32_i32 : i32
          %61 = arith.addi %51, %c32_i32 : i32
          %62 = arith.addi %54, %12 : index
          %63 = arith.addi %55, %12 : index
          %64 = arith.muli %56, %c64 : index
          %65 = arith.muli %57, %c64 : index
          %66 = arith.index_cast %58 : i32 to index
          %67 = arith.index_cast %59 : i32 to index
          %68 = arith.addi %64, %12 : index
          %69 = arith.addi %65, %12 : index
          %70 = arith.muli %66, %c64 : index
          %71 = arith.muli %67, %c64 : index
          %72 = arith.addi %70, %12 : index
          %73 = arith.addi %71, %12 : index
          %alloc_16 = memref.alloc() : memref<32x64xf16>
          %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%53], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          memref.copy %reinterpret_cast_17, %alloc_16 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %74 = bufferization.to_tensor %alloc_16 restrict writable : memref<32x64xf16>
          %transposed = linalg.transpose ins(%74 : tensor<32x64xf16>) outs(%cst_12 : tensor<64x32xf16>) permutation = [1, 0] 
          %75 = linalg.matmul {input_precision = "ieee"} ins(%30, %transposed : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%75 : tensor<32x32xf32>) outs(%alloc : memref<16x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
          %alloc_18 = memref.alloc() : memref<32x64xf16>
          %reinterpret_cast_19 = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          memref.copy %reinterpret_cast_19, %alloc_18 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %76 = bufferization.to_tensor %alloc_18 restrict writable : memref<32x64xf16>
          %transposed_20 = linalg.transpose ins(%76 : tensor<32x64xf16>) outs(%cst_13 : tensor<64x32xf16>) permutation = [1, 0] 
          %77 = linalg.matmul {input_precision = "ieee"} ins(%30, %transposed_20 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%77 : tensor<32x32xf32>) outs(%alloc_3 : memref<16x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %alloc_21 = memref.alloc() : memref<32x64xf16>
          %reinterpret_cast_22 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          memref.copy %reinterpret_cast_22, %alloc_21 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %78 = bufferization.to_tensor %alloc_21 restrict writable : memref<32x64xf16>
          %transposed_23 = linalg.transpose ins(%78 : tensor<32x64xf16>) outs(%cst_14 : tensor<64x32xf16>) permutation = [1, 0] 
          %79 = linalg.matmul {input_precision = "ieee"} ins(%30, %transposed_23 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%79 : tensor<32x32xf32>) outs(%alloc : memref<16x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %alloc_24 = memref.alloc() : memref<32x64xf16>
          %reinterpret_cast_25 = memref.reinterpret_cast %arg3 to offset: [%73], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          memref.copy %reinterpret_cast_25, %alloc_24 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %80 = bufferization.to_tensor %alloc_24 restrict writable : memref<32x64xf16>
          %transposed_26 = linalg.transpose ins(%80 : tensor<32x64xf16>) outs(%cst_15 : tensor<64x32xf16>) permutation = [1, 0] 
          %81 = linalg.matmul {input_precision = "ieee"} ins(%30, %transposed_26 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%81 : tensor<32x32xf32>) outs(%alloc_3 : memref<16x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %cst_27 = arith.constant dense<0.000000e+00> : tensor<32xf32>
          %cst_28 = arith.constant dense<[32, 2, 16]> : tensor<3xi64>
          %cst_29 = arith.constant dense<[2, 2, 16, 16]> : tensor<4xi64>
          %cst_30 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
          %alloc_31 = memref.alloc() : memref<32x64xf16>
          %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          memref.copy %reinterpret_cast_32, %alloc_31 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %82 = bufferization.to_tensor %alloc_31 restrict writable : memref<32x64xf16>
          %83 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<32x32xf16>
          %84 = linalg.matmul {input_precision = "ieee"} ins(%83, %82 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%cst_30 : tensor<32x64xf32>) -> tensor<32x64xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%84 : tensor<32x64xf32>) outs(%alloc_4 : memref<16x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %cst_33 = arith.constant dense<[32, 2, 16]> : tensor<3xi64>
          %cst_34 = arith.constant dense<[2, 2, 16, 16]> : tensor<4xi64>
          %cst_35 = arith.constant dense<[32, 2, 16]> : tensor<3xi64>
          %cst_36 = arith.constant dense<[2, 2, 16, 16]> : tensor<4xi64>
          %cst_37 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          %alloc_38 = memref.alloc() : memref<32x64xf16>
          %reinterpret_cast_39 = memref.reinterpret_cast %arg4 to offset: [%62], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          memref.copy %reinterpret_cast_39, %alloc_38 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %85 = bufferization.to_tensor %alloc_38 restrict writable : memref<32x64xf16>
          %86 = bufferization.to_tensor %memspacecast_10 restrict writable : memref<32x32xf16>
          %87 = linalg.matmul {input_precision = "ieee"} ins(%86, %85 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%cst_37 : tensor<32x64xf32>) -> tensor<32x64xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%87 : tensor<32x64xf32>) outs(%alloc_5 : memref<16x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 5
          %cst_40 = arith.constant dense<[32, 2, 16]> : tensor<3xi64>
          %cst_41 = arith.constant dense<[2, 2, 16, 16]> : tensor<4xi64>
          %cst_42 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          %alloc_43 = memref.alloc() : memref<32x64xf16>
          %reinterpret_cast_44 = memref.reinterpret_cast %arg4 to offset: [%68], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          memref.copy %reinterpret_cast_44, %alloc_43 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %88 = bufferization.to_tensor %alloc_43 restrict writable : memref<32x64xf16>
          %89 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<32x32xf16>
          %90 = linalg.matmul {input_precision = "ieee"} ins(%89, %88 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%cst_42 : tensor<32x64xf32>) -> tensor<32x64xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%90 : tensor<32x64xf32>) outs(%alloc_4 : memref<16x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %cst_45 = arith.constant dense<0.000000e+00> : tensor<32xf32>
          %cst_46 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          %alloc_47 = memref.alloc() : memref<32x64xf16>
          %reinterpret_cast_48 = memref.reinterpret_cast %arg4 to offset: [%72], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          memref.copy %reinterpret_cast_48, %alloc_47 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %91 = bufferization.to_tensor %alloc_47 restrict writable : memref<32x64xf16>
          %92 = bufferization.to_tensor %memspacecast_10 restrict writable : memref<32x32xf16>
          %93 = linalg.matmul {input_precision = "ieee"} ins(%92, %91 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%cst_46 : tensor<32x64xf32>) -> tensor<32x64xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%93 : tensor<32x64xf32>) outs(%alloc_5 : memref<16x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 7
          %cst_49 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
          scf.yield %cst_45, %cst_49, %cst_27, %60, %61 : tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32
        } {tt.divisibility_arg1 = dense<32> : tensor<1xi32>}
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
      %18 = tensor.empty() : tensor<16xf32>
      %19 = tensor.empty() : tensor<16x32xf32>
      %20 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<16x32xf32>) -> tensor<16x32xf32>
      %21 = tensor.empty() : tensor<16xf32>
      %22 = linalg.fill ins(%cst_0 : f32) outs(%21 : tensor<16xf32>) -> tensor<16xf32>
      %23 = tensor.empty() : tensor<16x32xf32>
      %24 = tensor.empty() : tensor<16x64xf32>
      %25 = tensor.empty() : tensor<16xf32>
      %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<16xf32>) -> tensor<16xf32>
      %27 = tensor.empty() : tensor<16x64xf32>
      %28 = linalg.fill ins(%cst_2 : f32) outs(%27 : tensor<16x64xf32>) -> tensor<16x64xf32>
      scope.scope : () -> () {
        %29 = hivm.hir.get_sub_block_idx -> i64
        %30 = arith.index_cast %29 : i64 to index
        %31:5 = scf.for %arg14 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg15 = %26, %arg16 = %28, %arg17 = %22, %arg18 = %c0_i32, %arg19 = %c0_i32) -> (tensor<16xf32>, tensor<16x64xf32>, tensor<16xf32>, i32, i32)  : i32 {
          %44 = arith.maxsi %arg18, %c0_i32 : i32
          %45 = arith.maxsi %arg19, %c0_i32 : i32
          %46 = tensor.empty() : tensor<64x32xf16>
          %47 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<16xf32>) -> tensor<16xf32>
          %48 = arith.addi %arg18, %c32_i32 : i32
          %49 = arith.addi %arg19, %c32_i32 : i32
          %50 = tensor.empty() : tensor<64x32xf16>
          %51 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<16xf32>) -> tensor<16xf32>
          %52 = tensor.empty() : tensor<64x32xf16>
          %53 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<16xf32>) -> tensor<16xf32>
          %54 = tensor.empty() : tensor<64x32xf16>
          %55 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<16xf32>) -> tensor<16xf32>
          %56 = arith.index_cast %44 : i32 to index
          %57 = arith.index_cast %45 : i32 to index
          %58 = arith.maxsi %48, %c0_i32 : i32
          %59 = arith.maxsi %49, %c0_i32 : i32
          %60 = arith.addi %48, %c32_i32 : i32
          %61 = arith.addi %49, %c32_i32 : i32
          %62 = arith.muli %56, %c64 : index
          %63 = arith.muli %57, %c64 : index
          %64 = arith.index_cast %58 : i32 to index
          %65 = arith.index_cast %59 : i32 to index
          %66 = arith.maxsi %60, %c0_i32 : i32
          %67 = arith.maxsi %61, %c0_i32 : i32
          %68 = arith.addi %60, %c32_i32 : i32
          %69 = arith.addi %61, %c32_i32 : i32
          %70 = arith.addi %62, %12 : index
          %71 = arith.addi %63, %12 : index
          %72 = arith.muli %64, %c64 : index
          %73 = arith.muli %65, %c64 : index
          %74 = arith.index_cast %66 : i32 to index
          %75 = arith.index_cast %67 : i32 to index
          %76 = arith.maxsi %68, %c0_i32 : i32
          %77 = arith.maxsi %69, %c0_i32 : i32
          %78 = arith.addi %68, %c32_i32 : i32
          %79 = arith.addi %69, %c32_i32 : i32
          %80 = arith.addi %72, %12 : index
          %81 = arith.addi %73, %12 : index
          %82 = arith.muli %74, %c64 : index
          %83 = arith.muli %75, %c64 : index
          %84 = arith.index_cast %76 : i32 to index
          %85 = arith.index_cast %77 : i32 to index
          %86 = arith.addi %82, %12 : index
          %87 = arith.addi %83, %12 : index
          %88 = arith.muli %84, %c64 : index
          %89 = arith.muli %85, %c64 : index
          %90 = arith.addi %88, %12 : index
          %91 = arith.addi %89, %12 : index
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
          %memspacecast = memref.memory_space_cast %alloc : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
          %92 = bufferization.to_tensor %memspacecast restrict writable : memref<16x32xf32>
          %93 = arith.mulf %92, %20 : tensor<16x32xf32>
          %reduced = linalg.reduce ins(%93 : tensor<16x32xf32>) outs(%22 : tensor<16xf32>) dimensions = [1] 
            (%in: f32, %init: f32) {
              %160 = arith.maximumf %in, %init : f32
              linalg.yield %160 : f32
            }
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %memspacecast_11 = memref.memory_space_cast %alloc_3 : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
          %94 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<16x32xf32>
          %95 = arith.mulf %94, %20 : tensor<16x32xf32>
          %96 = arith.maximumf %arg17, %reduced : tensor<16xf32>
          %reduced_12 = linalg.reduce ins(%95 : tensor<16x32xf32>) outs(%22 : tensor<16xf32>) dimensions = [1] 
            (%in: f32, %init: f32) {
              %160 = arith.maximumf %in, %init : f32
              linalg.yield %160 : f32
            }
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %memspacecast_13 = memref.memory_space_cast %alloc : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
          %97 = bufferization.to_tensor %memspacecast_13 restrict writable : memref<16x32xf32>
          %98 = arith.mulf %97, %20 : tensor<16x32xf32>
          %broadcasted_14 = linalg.broadcast ins(%96 : tensor<16xf32>) outs(%23 : tensor<16x32xf32>) dimensions = [1] 
          %99 = arith.subf %arg17, %96 : tensor<16xf32>
          %100 = arith.maximumf %96, %reduced_12 : tensor<16xf32>
          %reduced_15 = linalg.reduce ins(%98 : tensor<16x32xf32>) outs(%22 : tensor<16xf32>) dimensions = [1] 
            (%in: f32, %init: f32) {
              %160 = arith.maximumf %in, %init : f32
              linalg.yield %160 : f32
            }
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast_16 = memref.memory_space_cast %alloc_3 : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
          %101 = bufferization.to_tensor %memspacecast_16 restrict writable : memref<16x32xf32>
          %102 = arith.mulf %101, %20 : tensor<16x32xf32>
          %103 = arith.subf %93, %broadcasted_14 : tensor<16x32xf32>
          %104 = math.exp %99 : tensor<16xf32>
          %broadcasted_17 = linalg.broadcast ins(%100 : tensor<16xf32>) outs(%23 : tensor<16x32xf32>) dimensions = [1] 
          %105 = arith.subf %96, %100 : tensor<16xf32>
          %106 = arith.maximumf %100, %reduced_15 : tensor<16xf32>
          %reduced_18 = linalg.reduce ins(%102 : tensor<16x32xf32>) outs(%22 : tensor<16xf32>) dimensions = [1] 
            (%in: f32, %init: f32) {
              %160 = arith.maximumf %in, %init : f32
              linalg.yield %160 : f32
            }
          %107 = math.exp %103 : tensor<16x32xf32>
          %108 = arith.mulf %arg15, %104 : tensor<16xf32>
          %broadcasted_19 = linalg.broadcast ins(%104 : tensor<16xf32>) outs(%24 : tensor<16x64xf32>) dimensions = [1] 
          %109 = arith.subf %95, %broadcasted_17 : tensor<16x32xf32>
          %110 = math.exp %105 : tensor<16xf32>
          %broadcasted_20 = linalg.broadcast ins(%106 : tensor<16xf32>) outs(%23 : tensor<16x32xf32>) dimensions = [1] 
          %111 = arith.subf %100, %106 : tensor<16xf32>
          %112 = arith.maximumf %106, %reduced_18 : tensor<16xf32>
          %113 = arith.truncf %107 : tensor<16x32xf32> to tensor<16x32xf16>
          %cst_21 = arith.constant dense<[32, 2, 16]> : tensor<3xi64>
          %114 = tensor.empty() : tensor<2x32x16xf16>
          %cst_22 = arith.constant dense<[2, 2, 16, 16]> : tensor<4xi64>
          %cst_23 = arith.constant dense<[16, 2, 16]> : tensor<3xi64>
          %reshape = tensor.reshape %113(%cst_23) : (tensor<16x32xf16>, tensor<3xi64>) -> tensor<16x2x16xf16>
          %115 = tensor.empty() : tensor<2x16x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<16x2x16xf16>) outs(%115 : tensor<2x16x16xf16>) permutation = [1, 0, 2] 
          %cst_24 = arith.constant dense<[2, 1, 16, 16]> : tensor<4xi64>
          %reshape_25 = tensor.reshape %transposed(%cst_24) : (tensor<2x16x16xf16>, tensor<4xi64>) -> tensor<2x1x16x16xf16>
          %116 = bufferization.to_memref %reshape_25 : memref<2x1x16x16xf16>
          %memspacecast_26 = memref.memory_space_cast %116 : memref<2x1x16x16xf16> to memref<2x1x16x16xf16, #hivm.address_space<ub>>
          %c1 = arith.constant 1 : index
          %117 = arith.muli %30, %c1 : index
          %subview = memref.subview %alloc_7[0, %117, 0, 0] [2, 1, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<2x1x16x16xf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%memspacecast_26 : memref<2x1x16x16xf16, #hivm.address_space<ub>>) outs(%subview : memref<2x1x16x16xf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
          %reduced_27 = linalg.reduce ins(%107 : tensor<16x32xf32>) outs(%47 : tensor<16xf32>) dimensions = [1] 
            (%in: f32, %init: f32) {
              %160 = arith.addf %in, %init : f32
              linalg.yield %160 : f32
            }
          %118 = arith.mulf %arg16, %broadcasted_19 : tensor<16x64xf32>
          %119 = math.exp %109 : tensor<16x32xf32>
          %broadcasted_28 = linalg.broadcast ins(%110 : tensor<16xf32>) outs(%24 : tensor<16x64xf32>) dimensions = [1] 
          %120 = arith.subf %98, %broadcasted_20 : tensor<16x32xf32>
          %121 = math.exp %111 : tensor<16xf32>
          %broadcasted_29 = linalg.broadcast ins(%112 : tensor<16xf32>) outs(%23 : tensor<16x32xf32>) dimensions = [1] 
          %122 = arith.subf %106, %112 : tensor<16xf32>
          %123 = arith.addf %108, %reduced_27 : tensor<16xf32>
          %cst_30 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %memspacecast_31 = memref.memory_space_cast %alloc_4 : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x64xf32>
          %124 = bufferization.to_tensor %memspacecast_31 restrict writable : memref<16x64xf32>
          %125 = arith.addf %124, %118 : tensor<16x64xf32>
          %126 = arith.truncf %119 : tensor<16x32xf32> to tensor<16x32xf16>
          %cst_32 = arith.constant dense<[32, 2, 16]> : tensor<3xi64>
          %127 = tensor.empty() : tensor<2x32x16xf16>
          %cst_33 = arith.constant dense<[2, 2, 16, 16]> : tensor<4xi64>
          %cst_34 = arith.constant dense<[16, 2, 16]> : tensor<3xi64>
          %reshape_35 = tensor.reshape %126(%cst_34) : (tensor<16x32xf16>, tensor<3xi64>) -> tensor<16x2x16xf16>
          %128 = tensor.empty() : tensor<2x16x16xf16>
          %transposed_36 = linalg.transpose ins(%reshape_35 : tensor<16x2x16xf16>) outs(%128 : tensor<2x16x16xf16>) permutation = [1, 0, 2] 
          %cst_37 = arith.constant dense<[2, 1, 16, 16]> : tensor<4xi64>
          %reshape_38 = tensor.reshape %transposed_36(%cst_37) : (tensor<2x16x16xf16>, tensor<4xi64>) -> tensor<2x1x16x16xf16>
          %129 = bufferization.to_memref %reshape_38 : memref<2x1x16x16xf16>
          %memspacecast_39 = memref.memory_space_cast %129 : memref<2x1x16x16xf16> to memref<2x1x16x16xf16, #hivm.address_space<ub>>
          %c1_40 = arith.constant 1 : index
          %130 = arith.muli %30, %c1_40 : index
          %subview_41 = memref.subview %alloc_8[0, %130, 0, 0] [2, 1, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<2x1x16x16xf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%memspacecast_39 : memref<2x1x16x16xf16, #hivm.address_space<ub>>) outs(%subview_41 : memref<2x1x16x16xf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          %reduced_42 = linalg.reduce ins(%119 : tensor<16x32xf32>) outs(%51 : tensor<16xf32>) dimensions = [1] 
            (%in: f32, %init: f32) {
              %160 = arith.addf %in, %init : f32
              linalg.yield %160 : f32
            }
          %131 = math.exp %120 : tensor<16x32xf32>
          %broadcasted_43 = linalg.broadcast ins(%121 : tensor<16xf32>) outs(%24 : tensor<16x64xf32>) dimensions = [1] 
          %132 = arith.subf %102, %broadcasted_29 : tensor<16x32xf32>
          %133 = math.exp %122 : tensor<16xf32>
          %134 = arith.mulf %123, %110 : tensor<16xf32>
          %135 = arith.mulf %125, %broadcasted_28 : tensor<16x64xf32>
          %136 = arith.truncf %131 : tensor<16x32xf32> to tensor<16x32xf16>
          %cst_44 = arith.constant dense<[32, 2, 16]> : tensor<3xi64>
          %137 = tensor.empty() : tensor<2x32x16xf16>
          %cst_45 = arith.constant dense<[2, 2, 16, 16]> : tensor<4xi64>
          %cst_46 = arith.constant dense<[16, 2, 16]> : tensor<3xi64>
          %reshape_47 = tensor.reshape %136(%cst_46) : (tensor<16x32xf16>, tensor<3xi64>) -> tensor<16x2x16xf16>
          %138 = tensor.empty() : tensor<2x16x16xf16>
          %transposed_48 = linalg.transpose ins(%reshape_47 : tensor<16x2x16xf16>) outs(%138 : tensor<2x16x16xf16>) permutation = [1, 0, 2] 
          %cst_49 = arith.constant dense<[2, 1, 16, 16]> : tensor<4xi64>
          %reshape_50 = tensor.reshape %transposed_48(%cst_49) : (tensor<2x16x16xf16>, tensor<4xi64>) -> tensor<2x1x16x16xf16>
          %139 = bufferization.to_memref %reshape_50 : memref<2x1x16x16xf16>
          %memspacecast_51 = memref.memory_space_cast %139 : memref<2x1x16x16xf16> to memref<2x1x16x16xf16, #hivm.address_space<ub>>
          %c1_52 = arith.constant 1 : index
          %140 = arith.muli %30, %c1_52 : index
          %subview_53 = memref.subview %alloc_7[0, %140, 0, 0] [2, 1, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<2x1x16x16xf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%memspacecast_51 : memref<2x1x16x16xf16, #hivm.address_space<ub>>) outs(%subview_53 : memref<2x1x16x16xf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          %reduced_54 = linalg.reduce ins(%131 : tensor<16x32xf32>) outs(%53 : tensor<16xf32>) dimensions = [1] 
            (%in: f32, %init: f32) {
              %160 = arith.addf %in, %init : f32
              linalg.yield %160 : f32
            }
          %141 = math.exp %132 : tensor<16x32xf32>
          %broadcasted_55 = linalg.broadcast ins(%133 : tensor<16xf32>) outs(%24 : tensor<16x64xf32>) dimensions = [1] 
          %142 = arith.addf %134, %reduced_42 : tensor<16xf32>
          %cst_56 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
          %memspacecast_57 = memref.memory_space_cast %alloc_5 : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x64xf32>
          %143 = bufferization.to_tensor %memspacecast_57 restrict writable : memref<16x64xf32>
          %144 = arith.addf %143, %135 : tensor<16x64xf32>
          %145 = arith.truncf %141 : tensor<16x32xf32> to tensor<16x32xf16>
          %cst_58 = arith.constant dense<[32, 2, 16]> : tensor<3xi64>
          %146 = tensor.empty() : tensor<2x32x16xf16>
          %cst_59 = arith.constant dense<[2, 2, 16, 16]> : tensor<4xi64>
          %cst_60 = arith.constant dense<[16, 2, 16]> : tensor<3xi64>
          %reshape_61 = tensor.reshape %145(%cst_60) : (tensor<16x32xf16>, tensor<3xi64>) -> tensor<16x2x16xf16>
          %147 = tensor.empty() : tensor<2x16x16xf16>
          %transposed_62 = linalg.transpose ins(%reshape_61 : tensor<16x2x16xf16>) outs(%147 : tensor<2x16x16xf16>) permutation = [1, 0, 2] 
          %cst_63 = arith.constant dense<[2, 1, 16, 16]> : tensor<4xi64>
          %reshape_64 = tensor.reshape %transposed_62(%cst_63) : (tensor<2x16x16xf16>, tensor<4xi64>) -> tensor<2x1x16x16xf16>
          %148 = bufferization.to_memref %reshape_64 : memref<2x1x16x16xf16>
          %memspacecast_65 = memref.memory_space_cast %148 : memref<2x1x16x16xf16> to memref<2x1x16x16xf16, #hivm.address_space<ub>>
          %c1_66 = arith.constant 1 : index
          %149 = arith.muli %30, %c1_66 : index
          %subview_67 = memref.subview %alloc_8[0, %149, 0, 0] [2, 1, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<2x1x16x16xf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%memspacecast_65 : memref<2x1x16x16xf16, #hivm.address_space<ub>>) outs(%subview_67 : memref<2x1x16x16xf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          %reduced_68 = linalg.reduce ins(%141 : tensor<16x32xf32>) outs(%55 : tensor<16xf32>) dimensions = [1] 
            (%in: f32, %init: f32) {
              %160 = arith.addf %in, %init : f32
              linalg.yield %160 : f32
            }
          %150 = arith.mulf %142, %121 : tensor<16xf32>
          %151 = arith.mulf %144, %broadcasted_43 : tensor<16x64xf32>
          %152 = arith.addf %150, %reduced_54 : tensor<16xf32>
          %cst_69 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %memspacecast_70 = memref.memory_space_cast %alloc_4 : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x64xf32>
          %153 = bufferization.to_tensor %memspacecast_70 restrict writable : memref<16x64xf32>
          %154 = arith.addf %153, %151 : tensor<16x64xf32>
          %155 = arith.mulf %152, %133 : tensor<16xf32>
          %156 = arith.mulf %154, %broadcasted_55 : tensor<16x64xf32>
          %157 = arith.addf %155, %reduced_68 : tensor<16xf32>
          %cst_71 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
          %memspacecast_72 = memref.memory_space_cast %alloc_5 : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x64xf32>
          %158 = bufferization.to_tensor %memspacecast_72 restrict writable : memref<16x64xf32>
          %159 = arith.addf %158, %156 : tensor<16x64xf32>
          scf.yield %157, %159, %112, %78, %79 : tensor<16xf32>, tensor<16x64xf32>, tensor<16xf32>, i32, i32
        } {tt.divisibility_arg1 = dense<32> : tensor<1xi32>}
        %broadcasted = linalg.broadcast ins(%31#0 : tensor<16xf32>) outs(%24 : tensor<16x64xf32>) dimensions = [1] 
        %32 = arith.divf %31#1, %broadcasted : tensor<16x64xf32>
        %33 = math.log %31#0 : tensor<16xf32>
        %34 = arith.addf %31#2, %33 : tensor<16xf32>
        %35 = arith.muli %8, %c8192_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = arith.index_cast %13 : i32 to index
        %38 = arith.addi %36, %37 : index
        %c16 = arith.constant 16 : index
        %39 = arith.muli %30, %c16 : index
        %40 = arith.addi %38, %39 : index
        %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%40], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
        bufferization.materialize_in_destination %34 in writable %reinterpret_cast_9 : (tensor<16xf32>, memref<16xf32, strided<[1], offset: ?>>) -> ()
        %41 = arith.truncf %32 : tensor<16x64xf32> to tensor<16x64xf16>
        %c1024 = arith.constant 1024 : index
        %42 = arith.muli %30, %c1024 : index
        %43 = arith.addi %17, %42 : index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%43], sizes: [16, 64], strides: [64, 1] : memref<?xf16> to memref<16x64xf16, strided<[64, 1], offset: ?>>
        bufferization.materialize_in_destination %41 in writable %reinterpret_cast_10 : (tensor<16x64xf16>, memref<16x64xf16, strided<[64, 1], offset: ?>>) -> ()
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
    }
    return
  }
}

