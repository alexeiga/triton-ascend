#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_precheck.mlir"
SINGLE_CORE_TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_classification_single_core.mlir"
MIXED_CORE_TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_classification_mixed_core.mlir"
FA_TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_scheduling_fa.mlir"
SCOPE_HOISTING_TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_scope_hoisting.mlir"
ROLLBACK_TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_transaction_rollback.mlir"
CANDIDATE_FALLBACK_TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_candidate_fallback.mlir"
NESTED_CANDIDATE_TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_nested_candidate.mlir"
VERBOSE=false

case "${1:-}" in
  "") ;;
  -v|--verbose) VERBOSE=true ;;
  -h|--help)
    echo "usage: $0 [--verbose]"
    exit 0
    ;;
  *)
    echo "error: unknown option: $1" >&2
    echo "usage: $0 [--verbose]" >&2
    exit 2
    ;;
esac

# Keep pass diagnostics visible, but hide the large IR dump sections emitted on
# stderr. The transformed IR on stdout is left untouched for FileCheck.
filter_ir_dumps() {
  awk '
    /^\[cv-split\] === (IR|FUNCTION IR)/ { in_dump = 1; next }
    in_dump && /^\[cv-split\] === END/ { in_dump = 0; next }
    !in_dump { print }
  '
}

show_log_if_verbose() {
  if $VERBOSE; then
    filter_ir_dumps <"$1" >&2
  fi
}

run_stdout_filecheck() {
  local input="$1"
  local unroll="$2"
  local log="$3"
  shift 3

  if "$OPT" "$input" \
      "--cv_split_scheduling=compile-on-910-95=true unroll-factor=$unroll" \
      2>"$log" | "$FC" "$input" "$@"; then
    show_log_if_verbose "$log"
    return 0
  fi

  echo "    triton-opt diagnostics:" >&2
  filter_ir_dumps <"$log" >&2
  return 1
}

find_triton_opt() {
  if [[ -n "${TRITON_OPT:-}" ]]; then
    printf '%s\n' "$TRITON_OPT"
    return
  fi

  local vscode_debug="$REPO/build-vscode-debug/bin/triton-opt"
  if [[ -x "$vscode_debug" ]]; then
    printf '%s\n' "$vscode_debug"
    return
  fi

  local candidate
  for candidate in "$REPO"/python/build/cmake.linux-*-cpython-*/bin/triton-opt; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

find_filecheck() {
  if [[ -n "${FILECHECK:-}" ]]; then
    printf '%s\n' "$FILECHECK"
    return
  fi

  if [[ -n "${LLVM_SYSPATH:-}" && -x "$LLVM_SYSPATH/bin/FileCheck" ]]; then
    printf '%s\n' "$LLVM_SYSPATH/bin/FileCheck"
    return
  fi

  if command -v FileCheck >/dev/null 2>&1; then
    command -v FileCheck
    return
  fi

  local adjacent_llvm="$REPO/../llvm-project/build/bin/FileCheck"
  if [[ -x "$adjacent_llvm" ]]; then
    printf '%s\n' "$adjacent_llvm"
    return
  fi

  return 1
}

OPT="$(find_triton_opt)" || {
  echo "error: triton-opt not found; rebuild it or set TRITON_OPT" >&2
  exit 1
}

FC="$(find_filecheck)" || {
  echo "error: FileCheck not found; set FILECHECK or LLVM_SYSPATH" >&2
  exit 1
}

echo ">>> CVSplit precheck lit tests"
echo "    triton-opt: $OPT"
echo "    FileCheck:  $FC"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

run_stdout_filecheck "$TEST" 4 "$TMP_DIR/reject.log" --check-prefix=REJECT
echo "    PASS: rejected unsupported loop structures without transformation"

run_stdout_filecheck "$TEST" 3 "$TMP_DIR/bad-unroll.log" \
  --check-prefix=BAD-UNROLL
echo "    PASS: rejected unsupported unroll factor"

ACCEPT_LOG="$TMP_DIR/accept.log"
if ! "$OPT" "$TEST" \
    "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" \
    >/dev/null 2>"$ACCEPT_LOG"; then
  echo "    triton-opt diagnostics:" >&2
  filter_ir_dumps <"$ACCEPT_LOG" >&2
  exit 1
fi
if ! "$FC" "$TEST" --check-prefix=ACCEPT <"$ACCEPT_LOG"; then
  echo "    triton-opt diagnostics:" >&2
  filter_ir_dumps <"$ACCEPT_LOG" >&2
  exit 1
fi
if ! "$FC" "$TEST" --check-prefix=DIAG <"$ACCEPT_LOG"; then
  echo "    triton-opt diagnostics:" >&2
  filter_ir_dumps <"$ACCEPT_LOG" >&2
  exit 1
fi
show_log_if_verbose "$ACCEPT_LOG"
echo "    PASS: accepted supported candidate-loop structures"

for factor in 2 8; do
  FACTOR_LOG="$TMP_DIR/factor-$factor.log"
  if ! "$OPT" "$TEST" \
      "--cv_split_scheduling=compile-on-910-95=true unroll-factor=$factor" \
      >/dev/null 2>"$FACTOR_LOG"; then
    echo "    triton-opt diagnostics:" >&2
    filter_ir_dumps <"$FACTOR_LOG" >&2
    exit 1
  fi
  if ! "$FC" "$TEST" --check-prefix="FACTOR$factor" <"$FACTOR_LOG"; then
    echo "    triton-opt diagnostics:" >&2
    filter_ir_dumps <"$FACTOR_LOG" >&2
    exit 1
  fi
  show_log_if_verbose "$FACTOR_LOG"
  echo "    PASS: accepted supported unroll factor $factor"
done

echo ">>> CVSplit classification lit tests"
for classification_test in "$SINGLE_CORE_TEST" "$MIXED_CORE_TEST"; do
  test_name="$(basename "$classification_test" .mlir)"
  run_stdout_filecheck "$classification_test" 4 \
    "$TMP_DIR/$test_name-ir.log" --check-prefix=IR

  classification_log="$TMP_DIR/$test_name-diag.log"
  if ! "$OPT" "$classification_test" \
      "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" \
      >/dev/null 2>"$classification_log"; then
    echo "    triton-opt diagnostics:" >&2
    filter_ir_dumps <"$classification_log" >&2
    exit 1
  fi
  if ! "$FC" "$classification_test" --check-prefix=DIAG \
      <"$classification_log"; then
    echo "    triton-opt diagnostics:" >&2
    filter_ir_dumps <"$classification_log" >&2
    exit 1
  fi
  show_log_if_verbose "$classification_log"
done
echo "    PASS: rejected single-core loops"
echo "    PASS: accepted mixed-core loop"

echo ">>> CVSplit transactional rollback lit test"
run_stdout_filecheck "$ROLLBACK_TEST" 4 "$TMP_DIR/rollback-ir.log" \
  --check-prefix=IR
ROLLBACK_LOG="$TMP_DIR/rollback-diag.log"
if ! "$OPT" "$ROLLBACK_TEST" \
    "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" \
    >/dev/null 2>"$ROLLBACK_LOG"; then
  echo "    triton-opt diagnostics:" >&2
  filter_ir_dumps <"$ROLLBACK_LOG" >&2
  exit 1
fi
if ! "$FC" "$ROLLBACK_TEST" --check-prefix=DIAG <"$ROLLBACK_LOG"; then
  echo "    triton-opt diagnostics:" >&2
  filter_ir_dumps <"$ROLLBACK_LOG" >&2
  exit 1
fi
show_log_if_verbose "$ROLLBACK_LOG"
echo "    PASS: restored original IR after a late Stage-8 failure"

echo ">>> CVSplit per-candidate fallback lit test"
run_stdout_filecheck "$CANDIDATE_FALLBACK_TEST" 4 \
  "$TMP_DIR/candidate-fallback-ir.log" --check-prefix=IR
CANDIDATE_FALLBACK_LOG="$TMP_DIR/candidate-fallback-diag.log"
if ! "$OPT" "$CANDIDATE_FALLBACK_TEST" \
    "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" \
    >/dev/null 2>"$CANDIDATE_FALLBACK_LOG"; then
  echo "    triton-opt diagnostics:" >&2
  filter_ir_dumps <"$CANDIDATE_FALLBACK_LOG" >&2
  exit 1
fi
if ! "$FC" "$CANDIDATE_FALLBACK_TEST" --check-prefix=DIAG \
    <"$CANDIDATE_FALLBACK_LOG"; then
  echo "    triton-opt diagnostics:" >&2
  filter_ir_dumps <"$CANDIDATE_FALLBACK_LOG" >&2
  exit 1
fi
show_log_if_verbose "$CANDIDATE_FALLBACK_LOG"
echo "    PASS: skipped a failed candidate and committed the next candidate"

echo ">>> CVSplit nested candidate lit test"
run_stdout_filecheck "$NESTED_CANDIDATE_TEST" 4 \
  "$TMP_DIR/nested-candidate.log"
echo "    PASS: transformed the single innermost loop inside an outer loop"

echo ">>> CVSplit full Flash Attention lit test"
run_stdout_filecheck "$FA_TEST" 4 "$TMP_DIR/fa.log"
echo "    PASS: full Flash Attention transformation"

echo ">>> CVSplit scope hoisting lit test"
run_stdout_filecheck "$SCOPE_HOISTING_TEST" 4 "$TMP_DIR/scope-hoisting.log"
echo "    PASS: hoisted invariant CUBE layout view"

echo ">>> all CVSplit lit tests passed"
