#!/usr/bin/env fish

# ======================
# Configuration variables
# ======================
set BUILD_DIR "build"
set SRC "kicktriu.cl"
set BASENAME (basename $SRC .cl)
set GPU "gfx1150"
set OPTIMIZE 2

# ======================
# Setup
# ======================
mkdir -p $BUILD_DIR
rm -rf $BUILD_DIR/*

set LL "$BUILD_DIR/$BASENAME.ll"
set S  "$BUILD_DIR/$BASENAME.s"

# ======================
# Compile to LLVM IR
# ======================
clang \
  -x cl \
  -cl-std=CL2.0 \
  -target amdgcn-amd-amdhsa \
  -mcpu=$GPU \
  -S \
  -O$OPTIMIZE \
  -emit-llvm \
  -DCOEF_OF_RESTITUTION=1 \
  -DG=1 \
  -DNUDGE=1 \
  -DBOUNCE=1 \
  -cl-std=CL2.0 \
  $SRC \
  -o $LL

# ======================
# Compile to ISA (.s)
# ======================
clang \
  -target amdgcn-amd-amdhsa \
  -mcpu=$GPU \
  -S \
  -O$OPTIMIZE \
  $LL \
  -mllvm -amdgpu-dce-in-ra \
  -o $S

echo "Generated:"
echo "  $LL"
echo "  $S"

python scan_meta.py "$S"