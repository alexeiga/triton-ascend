#include "ascend/include/CVSplitScheduling/CrossScopeTransfers.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir::triton::cv_split {
namespace {

// ============================================================================
// Stage 8: Insert cross-scope transfers and synchronization
//
// For each SSA value that crosses from CUBE→VECTOR or VECTOR→CUBE:
//   C→V: alloc UB buffer, insert fixpipe + sync_block_set/wait, replace uses
//   V→C: alloc L1 buffer (NZ), insert copy + sync_block_set/wait, replace uses
//
// This runs BEFORE scope separation so both scopes see the shared buffers.
// ============================================================================

struct CrossScopeTransfer {
  Value value;
  Operation *producer;
  SmallVector<Operation *> consumers;
  enum Direction { CUBE_TO_VECTOR, VECTOR_TO_CUBE } direction;
};

static SmallVector<CrossScopeTransfer> findCrossScopeValues(
    Block *body,
    const DenseMap<Operation *, EngineType> &classification) {
  SmallVector<CrossScopeTransfer> transfers;

  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    auto prodIt = classification.find(&op);
    if (prodIt == classification.end())
      continue;
    EngineType prodType = prodIt->second;

    // C→V: Only transfer results of linalg.matmul (QK and PV dot products)
    // V→C: Only transfer values that DIRECTLY feed into linalg.matmul as operands
    //       (these are the P values after softmax+cast)
    //
    // Reference pattern for K=4:
    //   4× QK matmul results (C→V, fixpipe, flags 0-3)
    //   4× P inputs to PV matmul (V→C, copy UB→L1, flags 4-7)
    //   4× PV matmul results (C→V, fixpipe, flags 8-11)

    if (prodType == EngineType::CUBE && isa<linalg::MatmulOp>(&op)) {
      // C→V: matmul result consumed by VECTOR ops
      for (Value result : op.getResults()) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        SmallVector<Operation *> crossUsers;
        for (Operation *user : result.getUsers()) {
          if (user->getBlock() != body) continue;
          if (isa<scf::YieldOp>(user)) continue;
          auto consIt = classification.find(user);
          if (consIt == classification.end()) continue;
          if (consIt->second == EngineType::VECTOR)
            crossUsers.push_back(user);
        }
        if (!crossUsers.empty())
          transfers.push_back({result, &op, crossUsers,
                               CrossScopeTransfer::CUBE_TO_VECTOR});
      }
    } else if (prodType == EngineType::VECTOR) {
      // V→C: only if a VECTOR result feeds linalg.matmul as LHS (operand 0) or RHS (operand 1)
      // NOT operand 2 (the output/accumulator init)
      for (Value result : op.getResults()) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        SmallVector<Operation *> crossUsers;
        for (Operation *user : result.getUsers()) {
          if (user->getBlock() != body) continue;
          if (!isa<linalg::MatmulOp>(user)) continue;
          auto consIt = classification.find(user);
          if (consIt == classification.end()) continue;
          if (consIt->second != EngineType::CUBE) continue;
          // Check it's operand 0 or 1 (LHS/RHS), not 2 (init/accumulator)
          for (unsigned i = 0; i < 2; ++i) {
            if (user->getOperand(i) == result) {
              crossUsers.push_back(user);
              break;
            }
          }
        }
        if (!crossUsers.empty())
          transfers.push_back({result, &op, crossUsers,
                               CrossScopeTransfer::VECTOR_TO_CUBE});
      }
    }
  }
  return transfers;
}

// Per-pass attribute bundle shared by the transfer emitters: the core-type and
// pipe attributes are constant for the whole pass, and `loop` is where the
// shared buffers are allocated (just before the inner loop).
struct TransferEmitContext {
  MLIRContext *ctx;
  Location loc;
  scf::ForOp loop;
  hivm::TCoreTypeAttr cubeCoreAttr;
  hivm::TCoreTypeAttr vecCoreAttr;
  hivm::PipeAttr pipeFixAttr;
  hivm::PipeAttr pipeVAttr;
  hivm::PipeAttr pipeMte3Attr;
  hivm::PipeAttr pipeMte1Attr;
};

// alloc + annotation.mark{effects=["write","read"]}. One shared buffer per
// transfer, like the reference FA kernel. (The target IR carries only
// `effects`; the hivm.tightly_coupled_buffer<N> attribute is intentionally
// omitted.)
static memref::AllocOp createAnnotatedAlloc(OpBuilder &builder, Location loc,
                                            MemRefType allocType) {
  auto allocOp = builder.create<memref::AllocOp>(loc, allocType);
  auto markOp = builder.create<annotation::MarkOp>(loc, allocOp.getResult());
  auto writeAttr = builder.getStringAttr("write");
  auto readAttr = builder.getStringAttr("read");
  markOp->setAttr("effects", builder.getArrayAttr({writeAttr, readAttr}));
  return allocOp;
}

// Depth-2 ping/pong buffer pool. Transfers that share an identical buffer type
// (e.g. all the unrolled qk_ub fixpipe targets, or all the P L1 packs) reuse a
// rotating set of `depth` physical allocations instead of one fresh buffer per
// unrolled stage. This mirrors the manual kernel's qk_ub_0/1, pv_ub_0/1,
// p_l1_0/1 double buffering and keeps peak UB/L1 bounded so unroll>=2 fits in
// the 248 KB UB. Reuse serializes stage i and stage i+depth on the same buffer
// (WAR), which BiShengIR's GraphSyncSolver covers via the existing
// sync_block_set/wait flags — exactly the depth-2 software pipeline the manual
// kernel uses.
struct PingPongPool {
  llvm::StringMap<SmallVector<memref::AllocOp, 2>> slots;
  llvm::StringMap<unsigned> useCount;
  // One hoisted ND view (memory_space_cast of convert_layout) per L1 P buffer,
  // matching the manual kernel which emits a single convert_layout per p_l1
  // buffer before the KV loop and reuses it (fresh to_tensor per matmul).
  // Multiple convert_layout views of the same cbuf buffer confuse BiShengIR's
  // L1 NZ tracking and yield a misaligned / zero-burst matmul operand load.
  llvm::DenseMap<Operation *, Value> ndView;
  unsigned depth = 2;

  // Allocation to use for the next transfer of `allocType`: a new physical
  // buffer only while the rotating set is smaller than `depth`, otherwise the
  // round-robin reuse. `builder`'s insertion point must already be set (before
  // the loop) for the create case.
  memref::AllocOp getOrCreate(OpBuilder &builder, Location loc,
                              MemRefType allocType) {
    std::string sig;
    llvm::raw_string_ostream os(sig);
    os << allocType;
    (void)os.str();
    auto &vec = slots[sig];
    unsigned slot = useCount[sig]++ % depth;
    if (slot < vec.size())
      return vec[slot];
    auto allocOp = createAnnotatedAlloc(builder, loc, allocType);
    vec.push_back(allocOp);
    return allocOp;
  }
};

// CUBE -> VECTOR: the matmul (L0C) result is fixpipe'd to a shared UB buffer,
// CUBE signals via sync_block_set, VECTOR waits and reads it back as a tensor.
// ROW_SPLIT: the UB buffer is half height (16 rows); the fixpipe sends 16 rows
// to each veccore's private UB so both veccores stay busy (2x throughput). The
// VECTOR scope is re-tiled to 16 rows per veccore in a later stage.
static void emitCubeToVectorTransfer(const TransferEmitContext &c,
                                     CrossScopeTransfer &xfer,
                                     RankedTensorType tensorType, int flagId,
                                     PingPongPool &pool) {
  Type elemType = tensorType.getElementType();
  ArrayRef<int64_t> shape = tensorType.getShape();

  OpBuilder builder(c.ctx);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);

  auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
  SmallVector<int64_t, 4> ubShape(shape.begin(), shape.end());
  bool rowSplit = (ubShape[0] % 2 == 0);
  if (rowSplit) ubShape[0] /= 2;
  auto allocType = MemRefType::get(ubShape, elemType, nullptr, ubAddrSpace);
  auto halfTensorType = RankedTensorType::get(ubShape, elemType);

  // Ping/pong shared alloc before the loop (depth-2 reuse across unrolled
  // stages instead of one buffer per transfer).
  builder.setInsertionPoint(c.loop);
  auto sharedAllocOp = pool.getOrCreate(builder, c.loc, allocType);

  // fixpipe after the producer (inside loop body) -> writes the shared buffer.
  builder.setInsertionPointAfter(xfer.producer);
  auto dmaModeAttr = hivm::FixpipeDMAModeAttr::get(c.ctx, hivm::FixpipeDMAMode::NZ2ND);
  auto dualDstAttr = hivm::FixpipeDualDstModeAttr::get(c.ctx,
      rowSplit ? hivm::FixpipeDualDstMode::ROW_SPLIT
               : hivm::FixpipeDualDstMode::NO_DUAL);
  builder.create<hivm::FixpipeOp>(c.loc, mlir::TypeRange{},
      xfer.value,                    // src (full 32-row tile from dot)
      sharedAllocOp.getResult(),     // dst (16-row shared UB alloc)
      mlir::ValueRange{}, dmaModeAttr,
      dualDstAttr, nullptr, nullptr, nullptr, mlir::ArrayAttr{}, nullptr);

  // CUBE signals VECTOR.
  builder.create<hivm::SyncBlockSetOp>(c.loc, c.cubeCoreAttr, c.pipeFixAttr, c.pipeVAttr, flagAttr);

  // Consumer side: wait + read the shared buffer back as a 16-row tensor.
  Operation *firstConsumer = xfer.consumers.front();
  for (auto *cons : xfer.consumers)
    if (cons->isBeforeInBlock(firstConsumer))
      firstConsumer = cons;
  builder.setInsertionPoint(firstConsumer);

  builder.create<hivm::SyncBlockWaitOp>(c.loc, c.vecCoreAttr, c.pipeFixAttr, c.pipeVAttr, flagAttr);

  auto plainMemrefType = MemRefType::get(ubShape, elemType);
  auto castOp = builder.create<memref::MemorySpaceCastOp>(c.loc, plainMemrefType, sharedAllocOp.getResult());
  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      c.loc, halfTensorType, castOp.getResult(), /*restrict=*/true, /*writable=*/true);

  for (auto *consumer : xfer.consumers)
    consumer->replaceUsesOfWith(xfer.value, toTensorOp.getResult());

  llvm::errs() << "[cv-split]   C→V transfer #" << flagId
               << ": " << xfer.producer->getName()
               << " → " << ubShape[0] << "x" << ubShape[1]
               << " UB buffer (" << (rowSplit ? "ROW_SPLIT" : "NO_DUAL") << ")\n";
}

// VECTOR -> CUBE: a softmax/cast result is NZ-packed and copied UB->L1 into a
// shared L1 buffer, VECTOR signals via sync_block_set, CUBE waits and reads it
// back through a convert_layout (NZ fractal -> ND view) for matmul consumption.
// NZ packing applies only when both dims are multiples of 16; otherwise the L1
// buffer keeps the flat [M, N] layout.
static void emitVectorToCubeTransfer(const TransferEmitContext &c,
                                     CrossScopeTransfer &xfer,
                                     RankedTensorType tensorType, int flagId,
                                     int markAllocIndex, PingPongPool &pool) {
  Type elemType = tensorType.getElementType();
  ArrayRef<int64_t> shape = tensorType.getShape();

  OpBuilder builder(c.ctx);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);

  int64_t M = shape[0];
  int64_t N = shape[1];
  auto l1AddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);

  // NZ-fractal L1 layout: ND [M, N] is stored as [N/16, M/16, 16, 16] (B16
  // fractal) when both dims are multiples of 16; otherwise fall back to flat.
  bool useNZ = (M % 16 == 0) && (N % 16 == 0);
  int64_t N16 = N / 16, M16 = M / 16;
  SmallVector<int64_t, 4> l1Shape =
      useNZ ? SmallVector<int64_t, 4>{N16, M16, 16, 16}
            : SmallVector<int64_t, 4>{M, N};
  auto l1AllocType = MemRefType::get(l1Shape, elemType, nullptr, l1AddrSpace);

  // Ping/pong shared L1 alloc before the loop (depth-2 reuse across unrolled
  // stages instead of one buffer per transfer).
  builder.setInsertionPoint(c.loop);
  auto sharedL1AllocOp = pool.getOrCreate(builder, c.loc, l1AllocType);

  // Inside loop body after producer: (NZ pack) -> to_memref -> cast -> copy.
  builder.setInsertionPointAfter(xfer.producer);
  auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
  Value packedTensor = xfer.value;
  SmallVector<int64_t, 4> srcShape =
      useNZ ? SmallVector<int64_t, 4>{N16, M16, 16, 16}
            : SmallVector<int64_t, 4>{M, N};

  if (useNZ) {
    // ND [M,N] -> NZ [N/16, M/16, 16, 16] via reshape -> transpose -> reshape.
    auto i64Ty = builder.getI64Type();
    auto s3Type = RankedTensorType::get({3}, i64Ty);
    auto s3Const = builder.create<arith::ConstantOp>(c.loc, s3Type,
        DenseElementsAttr::get(s3Type, ArrayRef<int64_t>{M, N16, 16}));
    auto resh1Type = RankedTensorType::get({M, N16, 16}, elemType);
    auto resh1 = builder.create<tensor::ReshapeOp>(c.loc, resh1Type,
        xfer.value, s3Const.getResult());
    auto emptyT = builder.create<tensor::EmptyOp>(c.loc,
        ArrayRef<int64_t>{N16, M, 16}, elemType);
    auto transp = builder.create<linalg::TransposeOp>(c.loc, resh1.getResult(),
        emptyT.getResult(), ArrayRef<int64_t>{1, 0, 2});
    auto s4Type = RankedTensorType::get({4}, i64Ty);
    auto s4Const = builder.create<arith::ConstantOp>(c.loc, s4Type,
        DenseElementsAttr::get(s4Type, ArrayRef<int64_t>{N16, M16, 16, 16}));
    auto nzTensorType = RankedTensorType::get({N16, M16, 16, 16}, elemType);
    auto resh2 = builder.create<tensor::ReshapeOp>(c.loc, nzTensorType,
        transp->getResult(0), s4Const.getResult());
    packedTensor = resh2.getResult();
  }

  auto srcMemrefType = MemRefType::get(srcShape, elemType);
  auto toMemrefOp = builder.create<bufferization::ToMemrefOp>(
      c.loc, srcMemrefType, packedTensor);
  auto ubMemrefType = MemRefType::get(srcShape, elemType, nullptr, ubAddrSpace);
  auto ubCastOp = builder.create<memref::MemorySpaceCastOp>(
      c.loc, ubMemrefType, toMemrefOp.getResult());

  // UB -> L1 copy (same NZ/flat shape on both sides).
  builder.create<hivm::CopyOp>(c.loc, mlir::TypeRange{},
      ubCastOp.getResult(),           // src (UB memref)
      sharedL1AllocOp.getResult());   // dst (shared L1 memref)

  // VECTOR signals CUBE.
  builder.create<hivm::SyncBlockSetOp>(c.loc, c.vecCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr, flagAttr);

  // Consumer (CUBE) side: wait + convert_layout (NZ -> ND view) for matmul.
  Operation *firstConsumer = xfer.consumers.front();
  for (auto *cons : xfer.consumers)
    if (cons->isBeforeInBlock(firstConsumer))
      firstConsumer = cons;
  builder.setInsertionPoint(firstConsumer);

  builder.create<hivm::SyncBlockWaitOp>(c.loc, c.cubeCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr, flagAttr);

  // ONE convert_layout (NZ -> ND view) + memory_space_cast per shared L1
  // buffer, reused across the unrolled stages that share that buffer. The
  // manual kernel emits the convert_layout once per p_l1 buffer and only
  // re-reads it with a fresh to_tensor per matmul. Emitting a convert_layout
  // per unrolled stage produces several aliasing ND views of the same cbuf
  // buffer, which BiShengIR mis-tracks into a misaligned / zero-burst L1->L0
  // load. The view ops stay INSIDE the loop body (before the first consumer of
  // the first stage that uses this buffer) — hoisting them above the loop
  // breaks the MIX-kernel AIC/AIV split (SplitMixKernel can't get out-operands
  // for a scope-level convert_layout).
  Operation *l1Key = sharedL1AllocOp.getOperation();
  Value ndViewVal = pool.ndView.lookup(l1Key);
  if (!ndViewVal) {
    // Place at the FRONT of the loop body so the single view dominates every
    // consumer regardless of the order transfers are processed in (the reuse
    // for later stages must be dominated by this definition).
    OpBuilder viewBuilder(c.ctx);
    scf::ForOp loopMut = c.loop;
    viewBuilder.setInsertionPointToStart(loopMut.getBody());
    auto ndLayout = hivm::DataLayoutAttr::get(c.ctx, hivm::DataLayout::ND);
    auto ndL1Type = MemRefType::get(shape, elemType, nullptr, l1AddrSpace);
    auto convertOp = viewBuilder.create<hivm::ConvertLayoutOp>(
        c.loc, ndL1Type, sharedL1AllocOp.getResult(), ndLayout, ndLayout,
        DenseI64ArrayAttr::get(c.ctx, shape), ValueRange{});
    auto plainMemrefType = MemRefType::get(shape, elemType);
    auto castOp = viewBuilder.create<memref::MemorySpaceCastOp>(
        c.loc, plainMemrefType, convertOp.getResult());
    ndViewVal = castOp.getResult();
    pool.ndView[l1Key] = ndViewVal;
  }
  // Fresh to_tensor per consumer group (after the wait), like the manual.
  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      c.loc, tensorType, ndViewVal, true, true);

  for (auto *consumer : xfer.consumers)
    consumer->replaceUsesOfWith(xfer.value, toTensorOp.getResult());

  llvm::errs() << "[cv-split]   V→C transfer #" << flagId
               << " (tightly_coupled=" << markAllocIndex << ")"
               << ": " << xfer.producer->getName()
               << " → " << M << "x" << N << " L1 buffer\n";
}

} // namespace

void insertCrossScopeTransfers(
    scf::ForOp loop,
    Block *body,
    const DenseMap<Operation *, EngineType> &classification) {

  MLIRContext *ctx = loop.getContext();
  Location loc = loop.getLoc();

  auto transfers = findCrossScopeValues(body, classification);
  if (transfers.empty()) {
    llvm::errs() << "[cv-split] No cross-scope transfers needed\n";
    return;
  }

  // Sort transfers for clean flag numbering: C→V QK first, then V→C P, then C→V PV
  // QK = C→V with smaller shape; PV = C→V with larger shape; P = V→C
  std::stable_sort(transfers.begin(), transfers.end(),
      [](const CrossScopeTransfer &a, const CrossScopeTransfer &b) {
        if (a.direction != b.direction)
          return a.direction == CrossScopeTransfer::CUBE_TO_VECTOR;
        if (a.direction == CrossScopeTransfer::CUBE_TO_VECTOR) {
          auto aType = dyn_cast<RankedTensorType>(a.value.getType());
          auto bType = dyn_cast<RankedTensorType>(b.value.getType());
          if (aType && bType) {
            int64_t aSize = aType.getNumElements();
            int64_t bSize = bType.getNumElements();
            if (aSize != bSize) return aSize < bSize;
          }
        }
        return false;
      });

  llvm::errs() << "[cv-split] Found " << transfers.size()
               << " cross-scope value transfers\n";

  TransferEmitContext ec{
      ctx, loc, loop,
      hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE),
      hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR),
      hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX),
      hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_V),
      hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE3),
      hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE1)};

  // Per-channel flag counters. C->V (PIPE_FIX/PIPE_V) and V->C (PIPE_MTE3/
  // PIPE_MTE1) are independent hardware sync channels (the HW key is the
  // (set_pipe, wait_pipe, event_id) triple), so each gets its own 0.. range.
  // The backend WAIT.INTRA.BLOCK intrinsic encodes the flag as a 4-bit
  // immediate (valid 0..15); a single shared counter overflowed it at unroll-8
  // (flags 0-23, "Cannot select" for >=16). Splitting per channel keeps
  // unroll-8 in range (C->V 0-15, V->C 0-7) without reusing a flag inside a
  // channel (never a set before its wait -> no set_flag hazard).
  int cvFlagCounter = 0;  // CUBE -> VECTOR (PIPE_FIX / PIPE_V)
  int vcFlagCounter = 0;  // VECTOR -> CUBE (PIPE_MTE3 / PIPE_MTE1)
  int markAllocIndex = 0; // ordinal of the shared buffer across all transfers

  // Depth-2 ping/pong pool shared by all transfers: same-typed buffers (all
  // unrolled qk_ub, all pv_ub, all P L1) rotate over 2 physical allocations.
  PingPongPool pool;

  for (auto &xfer : transfers) {
    auto tensorType = dyn_cast<RankedTensorType>(xfer.value.getType());
    if (!tensorType) {
      llvm::errs() << "[cv-split]   Skipping non-tensor transfer: "
                   << xfer.value.getType() << "\n";
      continue;
    }
    if (tensorType.getRank() < 2) {
      llvm::errs() << "[cv-split]   Skipping rank-" << tensorType.getRank()
                   << " tensor\n";
      continue;
    }

    if (xfer.direction == CrossScopeTransfer::CUBE_TO_VECTOR) {
      emitCubeToVectorTransfer(ec, xfer, tensorType, cvFlagCounter++, pool);
    } else {
      emitVectorToCubeTransfer(ec, xfer, tensorType, vcFlagCounter++,
                               markAllocIndex, pool);
    }
    ++markAllocIndex;
  }

  llvm::errs() << "[cv-split] Inserted " << transfers.size()
               << " transfers with " << cvFlagCounter << " C->V + "
               << vcFlagCounter << " V->C sync flags, "
               << markAllocIndex << " tightly-coupled pairs\n";
}

} // namespace mlir::triton::cv_split
