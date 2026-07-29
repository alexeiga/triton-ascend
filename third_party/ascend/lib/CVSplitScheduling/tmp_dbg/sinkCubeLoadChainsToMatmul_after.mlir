"scope.scope"() ({
  %232 = "hivm.hir.convert_layout"(%55) <{dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>, static_output_shape = array<i64: 32, 32>}> {ssbuffer.core_type = "CUBE"} : (memref<2x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x32xf16, #hivm.address_space<cbuf>>
  %233 = "hivm.hir.convert_layout"(%53) <{dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>, static_output_shape = array<i64: 32, 32>}> {ssbuffer.core_type = "CUBE"} : (memref<2x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x32xf16, #hivm.address_space<cbuf>>
  %234 = "memref.memory_space_cast"(%232) {ssbuffer.core_type = "CUBE"} : (memref<32x32xf16, #hivm.address_space<cbuf>>) -> memref<32x32xf16>
  %235 = "memref.memory_space_cast"(%233) {ssbuffer.core_type = "CUBE"} : (memref<32x32xf16, #hivm.address_space<cbuf>>) -> memref<32x32xf16>
  %236:2 = "scf.for"(%12, %13, %50, %12, %12) ({
  ^bb0(%arg70: i32, %arg71: i32, %arg72: i32):
    %237 = "arith.maxsi"(%arg71, %12) {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %238 = "arith.maxsi"(%arg72, %12) {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %239 = "tensor.empty"() {ssbuffer.core_type = "CUBE"} : () -> tensor<64x32xf16>
    %240 = "arith.addi"(%arg71, %6) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %241 = "arith.addi"(%arg72, %6) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %242 = "tensor.empty"() {ssbuffer.core_type = "CUBE"} : () -> tensor<64x32xf16>
    %243 = "tensor.empty"() {ssbuffer.core_type = "CUBE"} : () -> tensor<64x32xf16>
    %244 = "tensor.empty"() {ssbuffer.core_type = "CUBE"} : () -> tensor<64x32xf16>
    %245 = "arith.index_cast"(%237) {ssbuffer.core_type = "CUBE"} : (i32) -> index
    %246 = "arith.index_cast"(%238) {ssbuffer.core_type = "CUBE"} : (i32) -> index
    %247 = "arith.maxsi"(%240, %12) {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %248 = "arith.maxsi"(%241, %12) {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %249 = "arith.addi"(%240, %6) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %250 = "arith.addi"(%241, %6) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %251 = "arith.muli"(%245, %1) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %252 = "arith.muli"(%246, %1) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %253 = "arith.index_cast"(%247) {ssbuffer.core_type = "CUBE"} : (i32) -> index
    %254 = "arith.index_cast"(%248) {ssbuffer.core_type = "CUBE"} : (i32) -> index
    %255 = "arith.maxsi"(%249, %12) {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %256 = "arith.maxsi"(%250, %12) {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %257 = "arith.addi"(%249, %6) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %258 = "arith.addi"(%250, %6) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %259 = "arith.addi"(%251, %35) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %260 = "arith.addi"(%252, %35) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %261 = "arith.muli"(%253, %1) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %262 = "arith.muli"(%254, %1) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %263 = "arith.index_cast"(%255) {ssbuffer.core_type = "CUBE"} : (i32) -> index
    %264 = "arith.index_cast"(%256) {ssbuffer.core_type = "CUBE"} : (i32) -> index
    %265 = "arith.maxsi"(%257, %12) {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %266 = "arith.maxsi"(%258, %12) {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %267 = "arith.addi"(%257, %6) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %268 = "arith.addi"(%258, %6) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (i32, i32) -> i32
    %269 = "arith.addi"(%261, %35) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %270 = "arith.addi"(%262, %35) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %271 = "arith.muli"(%263, %1) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %272 = "arith.muli"(%264, %1) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %273 = "arith.index_cast"(%265) {ssbuffer.core_type = "CUBE"} : (i32) -> index
    %274 = "arith.index_cast"(%266) {ssbuffer.core_type = "CUBE"} : (i32) -> index
    %275 = "arith.addi"(%271, %35) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %276 = "arith.addi"(%272, %35) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %277 = "arith.muli"(%273, %1) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %278 = "arith.muli"(%274, %1) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %279 = "arith.addi"(%277, %35) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %280 = "arith.addi"(%278, %35) <{overflowFlags = #arith.overflow<none>}> {ssbuffer.core_type = "CUBE"} : (index, index) -> index
    %281 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.core_type = "CUBE"} : () -> memref<32x64xf16>
    %282 = "memref.reinterpret_cast"(%arg3, %260) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> {ssbuffer.core_type = "CUBE"} : (memref<?xf16>, index) -> memref<32x64xf16, strided<[64, 1], offset: ?>>
    "memref.copy"(%282, %281) {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16, strided<[64, 1], offset: ?>>, memref<32x64xf16>) -> ()
    %283 = "bufferization.to_tensor"(%281) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16>) -> tensor<32x64xf16>
    %284 = "linalg.transpose"(%283, %239) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%arg103: f16, %arg104: f16):
      "linalg.yield"(%arg103) : (f16) -> ()
    }) {ssbuffer.core_type = "CUBE"} : (tensor<32x64xf16>, tensor<64x32xf16>) -> tensor<64x32xf16>
    %285 = "linalg.matmul"(%49, %284, %22) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg100: f16, %arg101: f16, %arg102: f32):
      %364 = "arith.extf"(%arg100) : (f16) -> f32
      %365 = "arith.extf"(%arg101) : (f16) -> f32
      %366 = "arith.mulf"(%364, %365) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %367 = "arith.addf"(%arg102, %366) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%367) : (f32) -> ()
    }) {input_precision = "ieee", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], ssbuffer.core_type = "CUBE"} : (tensor<32x64xf16>, tensor<64x32xf16>, tensor<32x32xf32>) -> tensor<32x32xf32>
    "hivm.hir.fixpipe"(%285, %51) <{dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> {ssbuffer.core_type = "CUBE"} : (tensor<32x32xf32>, memref<16x32xf32, #hivm.address_space<ub>>) -> ()
    "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 0 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %286 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.core_type = "CUBE"} : () -> memref<32x64xf16>
    %287 = "memref.reinterpret_cast"(%arg3, %270) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> {ssbuffer.core_type = "CUBE"} : (memref<?xf16>, index) -> memref<32x64xf16, strided<[64, 1], offset: ?>>
    "memref.copy"(%287, %286) {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16, strided<[64, 1], offset: ?>>, memref<32x64xf16>) -> ()
    %288 = "bufferization.to_tensor"(%286) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16>) -> tensor<32x64xf16>
    %289 = "linalg.transpose"(%288, %242) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%arg98: f16, %arg99: f16):
      "linalg.yield"(%arg98) : (f16) -> ()
    }) {ssbuffer.core_type = "CUBE"} : (tensor<32x64xf16>, tensor<64x32xf16>) -> tensor<64x32xf16>
    %290 = "linalg.matmul"(%49, %289, %22) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg95: f16, %arg96: f16, %arg97: f32):
      %360 = "arith.extf"(%arg95) : (f16) -> f32
      %361 = "arith.extf"(%arg96) : (f16) -> f32
      %362 = "arith.mulf"(%360, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %363 = "arith.addf"(%arg97, %362) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%363) : (f32) -> ()
    }) {input_precision = "ieee", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], ssbuffer.core_type = "CUBE"} : (tensor<32x64xf16>, tensor<64x32xf16>, tensor<32x32xf32>) -> tensor<32x32xf32>
    "hivm.hir.fixpipe"(%290, %52) <{dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> {ssbuffer.core_type = "CUBE"} : (tensor<32x32xf32>, memref<16x32xf32, #hivm.address_space<ub>>) -> ()
    "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 1 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %291 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.core_type = "CUBE"} : () -> memref<32x64xf16>
    %292 = "memref.reinterpret_cast"(%arg3, %276) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> {ssbuffer.core_type = "CUBE"} : (memref<?xf16>, index) -> memref<32x64xf16, strided<[64, 1], offset: ?>>
    "memref.copy"(%292, %291) {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16, strided<[64, 1], offset: ?>>, memref<32x64xf16>) -> ()
    %293 = "bufferization.to_tensor"(%291) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16>) -> tensor<32x64xf16>
    %294 = "linalg.transpose"(%293, %243) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%arg93: f16, %arg94: f16):
      "linalg.yield"(%arg93) : (f16) -> ()
    }) {ssbuffer.core_type = "CUBE"} : (tensor<32x64xf16>, tensor<64x32xf16>) -> tensor<64x32xf16>
    %295 = "linalg.matmul"(%49, %294, %22) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg90: f16, %arg91: f16, %arg92: f32):
      %356 = "arith.extf"(%arg90) : (f16) -> f32
      %357 = "arith.extf"(%arg91) : (f16) -> f32
      %358 = "arith.mulf"(%356, %357) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %359 = "arith.addf"(%arg92, %358) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%359) : (f32) -> ()
    }) {input_precision = "ieee", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], ssbuffer.core_type = "CUBE"} : (tensor<32x64xf16>, tensor<64x32xf16>, tensor<32x32xf32>) -> tensor<32x32xf32>
    "hivm.hir.fixpipe"(%295, %51) <{dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> {ssbuffer.core_type = "CUBE"} : (tensor<32x32xf32>, memref<16x32xf32, #hivm.address_space<ub>>) -> ()
    "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 2 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %296 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.core_type = "CUBE"} : () -> memref<32x64xf16>
    %297 = "memref.reinterpret_cast"(%arg3, %280) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> {ssbuffer.core_type = "CUBE"} : (memref<?xf16>, index) -> memref<32x64xf16, strided<[64, 1], offset: ?>>
    "memref.copy"(%297, %296) {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16, strided<[64, 1], offset: ?>>, memref<32x64xf16>) -> ()
    %298 = "bufferization.to_tensor"(%296) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16>) -> tensor<32x64xf16>
    %299 = "linalg.transpose"(%298, %244) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%arg88: f16, %arg89: f16):
      "linalg.yield"(%arg88) : (f16) -> ()
    }) {ssbuffer.core_type = "CUBE"} : (tensor<32x64xf16>, tensor<64x32xf16>) -> tensor<64x32xf16>
    %300 = "linalg.matmul"(%49, %299, %22) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg85: f16, %arg86: f16, %arg87: f32):
      %352 = "arith.extf"(%arg85) : (f16) -> f32
      %353 = "arith.extf"(%arg86) : (f16) -> f32
      %354 = "arith.mulf"(%352, %353) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %355 = "arith.addf"(%arg87, %354) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%355) : (f32) -> ()
    }) {input_precision = "ieee", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], ssbuffer.core_type = "CUBE"} : (tensor<32x64xf16>, tensor<64x32xf16>, tensor<32x32xf32>) -> tensor<32x32xf32>
    "hivm.hir.fixpipe"(%300, %52) <{dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> {ssbuffer.core_type = "CUBE"} : (tensor<32x32xf32>, memref<16x32xf32, #hivm.address_space<ub>>) -> ()
    "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 3 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %301 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
    %302 = "arith.constant"() <{value = dense<[32, 2, 16]> : tensor<3xi64>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<3xi64>
    %303 = "arith.constant"() <{value = dense<[2, 2, 16, 16]> : tensor<4xi64>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<4xi64>
    %304 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<32x64xf32>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<32x64xf32>
    "hivm.hir.sync_block_wait"() <{pipe = #hivm.pipe<PIPE_MTE1>, static_flag_id = 4 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_MTE3>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %305 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.core_type = "CUBE"} : () -> memref<32x64xf16>
    %306 = "memref.reinterpret_cast"(%arg4, %259) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> {ssbuffer.core_type = "CUBE"} : (memref<?xf16>, index) -> memref<32x64xf16, strided<[64, 1], offset: ?>>
    "memref.copy"(%306, %305) {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16, strided<[64, 1], offset: ?>>, memref<32x64xf16>) -> ()
    %307 = "bufferization.to_tensor"(%305) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16>) -> tensor<32x64xf16>
    %308 = "bufferization.to_tensor"(%235) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x32xf16>) -> tensor<32x32xf16>
    %309 = "linalg.matmul"(%308, %307, %304) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg82: f16, %arg83: f16, %arg84: f32):
      %348 = "arith.extf"(%arg82) : (f16) -> f32
      %349 = "arith.extf"(%arg83) : (f16) -> f32
      %350 = "arith.mulf"(%348, %349) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %351 = "arith.addf"(%arg84, %350) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%351) : (f32) -> ()
    }) {input_precision = "ieee", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], ssbuffer.core_type = "CUBE"} : (tensor<32x32xf16>, tensor<32x64xf16>, tensor<32x64xf32>) -> tensor<32x64xf32>
    "hivm.hir.fixpipe"(%309, %54) <{dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> {ssbuffer.core_type = "CUBE"} : (tensor<32x64xf32>, memref<16x64xf32, #hivm.address_space<ub>>) -> ()
    "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 5 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %310 = "arith.constant"() <{value = dense<[32, 2, 16]> : tensor<3xi64>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<3xi64>
    %311 = "arith.constant"() <{value = dense<[2, 2, 16, 16]> : tensor<4xi64>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<4xi64>
    %312 = "arith.constant"() <{value = dense<[32, 2, 16]> : tensor<3xi64>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<3xi64>
    %313 = "arith.constant"() <{value = dense<[2, 2, 16, 16]> : tensor<4xi64>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<4xi64>
    %314 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<32x64xf32>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<32x64xf32>
    "hivm.hir.sync_block_wait"() <{pipe = #hivm.pipe<PIPE_MTE1>, static_flag_id = 6 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_MTE3>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %315 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.core_type = "CUBE"} : () -> memref<32x64xf16>
    %316 = "memref.reinterpret_cast"(%arg4, %269) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> {ssbuffer.core_type = "CUBE"} : (memref<?xf16>, index) -> memref<32x64xf16, strided<[64, 1], offset: ?>>
    "memref.copy"(%316, %315) {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16, strided<[64, 1], offset: ?>>, memref<32x64xf16>) -> ()
    %317 = "bufferization.to_tensor"(%315) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16>) -> tensor<32x64xf16>
    %318 = "bufferization.to_tensor"(%234) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x32xf16>) -> tensor<32x32xf16>
    %319 = "linalg.matmul"(%318, %317, %314) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg79: f16, %arg80: f16, %arg81: f32):
      %344 = "arith.extf"(%arg79) : (f16) -> f32
      %345 = "arith.extf"(%arg80) : (f16) -> f32
      %346 = "arith.mulf"(%344, %345) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %347 = "arith.addf"(%arg81, %346) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%347) : (f32) -> ()
    }) {input_precision = "ieee", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], ssbuffer.core_type = "CUBE"} : (tensor<32x32xf16>, tensor<32x64xf16>, tensor<32x64xf32>) -> tensor<32x64xf32>
    "hivm.hir.fixpipe"(%319, %56) <{dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> {ssbuffer.core_type = "CUBE"} : (tensor<32x64xf32>, memref<16x64xf32, #hivm.address_space<ub>>) -> ()
    "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 8 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %320 = "arith.constant"() <{value = dense<[32, 2, 16]> : tensor<3xi64>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<3xi64>
    %321 = "arith.constant"() <{value = dense<[2, 2, 16, 16]> : tensor<4xi64>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<4xi64>
    %322 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<32x64xf32>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<32x64xf32>
    "hivm.hir.sync_block_wait"() <{pipe = #hivm.pipe<PIPE_MTE1>, static_flag_id = 7 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_MTE3>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %323 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.core_type = "CUBE"} : () -> memref<32x64xf16>
    %324 = "memref.reinterpret_cast"(%arg4, %275) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> {ssbuffer.core_type = "CUBE"} : (memref<?xf16>, index) -> memref<32x64xf16, strided<[64, 1], offset: ?>>
    "memref.copy"(%324, %323) {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16, strided<[64, 1], offset: ?>>, memref<32x64xf16>) -> ()
    %325 = "bufferization.to_tensor"(%323) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16>) -> tensor<32x64xf16>
    %326 = "bufferization.to_tensor"(%235) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x32xf16>) -> tensor<32x32xf16>
    %327 = "linalg.matmul"(%326, %325, %322) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg76: f16, %arg77: f16, %arg78: f32):
      %340 = "arith.extf"(%arg76) : (f16) -> f32
      %341 = "arith.extf"(%arg77) : (f16) -> f32
      %342 = "arith.mulf"(%340, %341) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %343 = "arith.addf"(%arg78, %342) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%343) : (f32) -> ()
    }) {input_precision = "ieee", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], ssbuffer.core_type = "CUBE"} : (tensor<32x32xf16>, tensor<32x64xf16>, tensor<32x64xf32>) -> tensor<32x64xf32>
    "hivm.hir.fixpipe"(%327, %54) <{dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> {ssbuffer.core_type = "CUBE"} : (tensor<32x64xf32>, memref<16x64xf32, #hivm.address_space<ub>>) -> ()
    "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 10 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %328 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
    %329 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<32x64xf32>}> {ssbuffer.core_type = "CUBE"} : () -> tensor<32x64xf32>
    "hivm.hir.sync_block_wait"() <{pipe = #hivm.pipe<PIPE_MTE1>, static_flag_id = 9 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_MTE3>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %330 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.core_type = "CUBE"} : () -> memref<32x64xf16>
    %331 = "memref.reinterpret_cast"(%arg4, %279) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> {ssbuffer.core_type = "CUBE"} : (memref<?xf16>, index) -> memref<32x64xf16, strided<[64, 1], offset: ?>>
    "memref.copy"(%331, %330) {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16, strided<[64, 1], offset: ?>>, memref<32x64xf16>) -> ()
    %332 = "bufferization.to_tensor"(%330) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x64xf16>) -> tensor<32x64xf16>
    %333 = "bufferization.to_tensor"(%234) <{restrict, writable}> {ssbuffer.core_type = "CUBE"} : (memref<32x32xf16>) -> tensor<32x32xf16>
    %334 = "linalg.matmul"(%333, %332, %329) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg73: f16, %arg74: f16, %arg75: f32):
      %336 = "arith.extf"(%arg73) : (f16) -> f32
      %337 = "arith.extf"(%arg74) : (f16) -> f32
      %338 = "arith.mulf"(%336, %337) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %339 = "arith.addf"(%arg75, %338) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%339) : (f32) -> ()
    }) {input_precision = "ieee", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], ssbuffer.core_type = "CUBE"} : (tensor<32x32xf16>, tensor<32x64xf16>, tensor<32x64xf32>) -> tensor<32x64xf32>
    "hivm.hir.fixpipe"(%334, %56) <{dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> {ssbuffer.core_type = "CUBE"} : (tensor<32x64xf32>, memref<16x64xf32, #hivm.address_space<ub>>) -> ()
    "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 11 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.core_type = "CUBE"} : () -> ()
    %335 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<32x64xf32>}> : () -> tensor<32x64xf32>
    "scf.yield"(%267, %268) : (i32, i32) -> ()
  }) {ssbuffer.core_type = "VECTOR, CUBE, VECTOR, CUBE, CUBE", tt.divisibility_arg1 = dense<32> : tensor<1xi32>} : (i32, i32, i32, i32, i32) -> (i32, i32)
  "scope.return"() : () -> ()
}) {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline} : () -> ()
