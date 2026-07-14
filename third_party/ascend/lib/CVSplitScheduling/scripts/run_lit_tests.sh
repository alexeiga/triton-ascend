#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
TEST="$REPO/third_party/ascend/unittest/Conversion/General/CVSplitScheduling/cv_split_precheck.mlir"

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

"$OPT" "$TEST" \
  "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" \
  2>/dev/null | "$FC" "$TEST" --check-prefix=REJECT
echo "    PASS: rejected unsupported loop structures without transformation"

"$OPT" "$TEST" \
  "--cv_split_scheduling=compile-on-910-95=true unroll-factor=3" \
  2>/dev/null | "$FC" "$TEST" --check-prefix=BAD-UNROLL
echo "    PASS: rejected unsupported unroll factor"

"$OPT" "$TEST" \
  "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" \
  2>&1 >/dev/null | "$FC" "$TEST" --check-prefix=ACCEPT
echo "    PASS: accepted supported candidate-loop structures"

echo ">>> all CVSplit precheck lit tests passed"
