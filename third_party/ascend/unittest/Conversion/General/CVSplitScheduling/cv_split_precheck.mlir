// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=REJECT
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=3" 2>/dev/null | FileCheck %s --check-prefix=BAD-UNROLL
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=ACCEPT

// Pre-check rejection must happen before unrolling or scope construction.
// REJECT-NOT: scope.scope
// REJECT-LABEL: func.func @no_loop
// REJECT-NEXT: return
func.func @no_loop() {
  return
}

// ACCEPT: [cv-split] Function: single_loop_without_store
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @single_loop_without_store(%lb: index, %ub: index, %step: index) {
  scf.for %iv = %lb to %ub step %step {
    arith.addi %iv, %step : index
  }
  return
}

// An enclosing loop is supported when it contains exactly one innermost
// candidate loop.
// ACCEPT: [cv-split] Function: one_nested_candidate
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @one_nested_candidate(%lb: index, %ub: index, %step: index) {
  scf.for %outer = %lb to %ub step %step {
    scf.for %inner = %lb to %ub step %step {
      arith.addi %inner, %step : index
    }
  }
  return
}

// A store in the enclosing loop is outside the selected inner candidate and
// must not make pre-check reject the function.
// ACCEPT: [cv-split] Function: outer_store_after_inner_candidate
// ACCEPT-NEXT: [cv-split] Pre-check accepted candidate loop
func.func @outer_store_after_inner_candidate(
    %lb: index, %ub: index, %step: index, %value: i32,
    %dst: memref<?xi32>) {
  scf.for %outer = %lb to %ub step %step {
    scf.for %inner = %lb to %ub step %step {
      arith.addi %inner, %step : index
    }
    memref.store %value, %dst[%outer] : memref<?xi32>
  }
  return
}

// REJECT-LABEL: func.func @multiple_innermost_loops
// REJECT: scf.for
// REJECT: scf.for
func.func @multiple_innermost_loops(%lb: index, %ub: index, %step: index) {
  scf.for %first = %lb to %ub step %step {
  }
  scf.for %second = %lb to %ub step %step {
  }
  return
}

// REJECT-LABEL: func.func @store_in_candidate
// REJECT: scf.for
// REJECT: memref.store
// BAD-UNROLL-LABEL: func.func @store_in_candidate
// BAD-UNROLL: scf.for
// BAD-UNROLL-NOT: scope.scope
func.func @store_in_candidate(%lb: index, %ub: index, %step: index,
                              %value: i32, %dst: memref<1xi32>) {
  scf.for %iv = %lb to %ub step %step {
    %c0 = arith.constant 0 : index
    memref.store %value, %dst[%c0] : memref<1xi32>
  }
  return
}
