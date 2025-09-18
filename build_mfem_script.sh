#!/usr/bin/env bash

# Variant-key MFEM/Hypre build script
# - Out-of-source builds for Hypre into per-variant prefixes
# - MFEM built against selected Hypre install
# - Metadata stamped per dependency

set -euo pipefail
IFS=$'\n\t'
trap 'echo "Error on line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

step () { local label="$1"; shift; echo -e "\n==> $label"; "$@"; echo "<== $label done"; }

# 1) Module setup (only in SLURM)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  module purge
  module load gcc/13.2.0.lua cmake/3.27.9-gcc-13.2.0.lua cuda/12.4.0-gcc-13.2.0.lua \
              python/3.12.1-gcc-13.2.0.lua openblas/0.3.26-gcc-13.2.0.lua openmpi/5.0.3-gcc-13.2.0.lua
fi

# 2) Inputs / defaults
WITH_MPI=${WITH_MPI:-1}
WITH_CUDA=${WITH_CUDA:-0}
GPU_AWARE=${GPU_AWARE:-1}
PRECISION=${PRECISION:-double}          # double|single
BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo}
SYMBOLS=${SYMBOLS:-1}

# Export knobs so inner shells (bash -c) can read them reliably
export WITH_MPI WITH_CUDA GPU_AWARE PRECISION BUILD_TYPE SYMBOLS

HYPRE_SRC=${HYPRE_SRC:-"$HOME/opt/src/hypre"}
MFEM_SRC=${MFEM_SRC:-"$HOME/opt/src/mfem"}
METIS_SRC=${METIS_SRC:-"$HOME/opt/src/metis-5.1.0"}

HYPRE_TAG=${HYPRE_TAG:-latest}          # latest|master|vX.Y.Z
MFEM_TAG=${MFEM_TAG:-latest}

DEPS_PREFIX_BASE=${DEPS_PREFIX_BASE:-"/scratch/ajn6-amg/multi_gpu"}
DEPS_BUILD_ROOT="$DEPS_PREFIX_BASE/build"
DEPS_INSTALL_ROOT="$DEPS_PREFIX_BASE/deps"
mkdir -p "$DEPS_BUILD_ROOT" "$DEPS_INSTALL_ROOT" logs

# 3) GPU detection for CUDA arch
if [[ "$WITH_CUDA" == 1 ]] && command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
  mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u)
  if ((${#caps[@]} == 1)); then
    cap="${caps[0]}"; major=${cap%%.*}; minor=${cap##*.}
    export GPU_CC="${major}${minor}"; export CUDA_ARCH="sm_${GPU_CC}"
    echo "Detected CUDA_ARCH=${CUDA_ARCH}"
  else
    echo "Mixed/no GPU compute capability detected: ${caps[*]:-none}. Set CUDA_ARCH manually or ensure homogeneous allocation." >&2
  fi
fi

# 4) Variant key
MPIK=$([[ "$WITH_MPI" == 1 ]] && echo mpi || echo seq)
CUDAK=$([[ "$WITH_CUDA" == 1 ]] && echo cuda || echo cpu)
GAK=$([[ "$GPU_AWARE" == 1 ]] && echo gawa || echo nogawa)
PCK=$([[ "$PRECISION" == single ]] && echo sp || echo dp)
BTK=${BUILD_TYPE}
VARIANT_KEY="${MPIK}-${CUDAK}-${GAK}-${PCK}-${BTK}"
echo "Variant key: $VARIANT_KEY"

H_BUILD="$DEPS_BUILD_ROOT/$VARIANT_KEY/hypre"
H_INST="$DEPS_INSTALL_ROOT/$VARIANT_KEY/hypre"
M_BUILD="$DEPS_BUILD_ROOT/$VARIANT_KEY/mfem"
M_INST="$DEPS_INSTALL_ROOT/$VARIANT_KEY/mfem"
mkdir -p "$H_BUILD" "$H_INST" "$M_BUILD" "$M_INST"

# 5) Checkout helpers
latest_tag() { git -C "$1" tag --list | sort -V | tail -n1; }
checkout_ref() {
  local repo="$1" ref="$2"
  step "Update $repo" bash -c "git -C '$repo' fetch --tags --all --prune || true"
  case "$ref" in
    latest) ref=$(latest_tag "$repo");;
    master) ref=master;;
  esac
  if [[ -z "$ref" ]]; then echo "No tag found for $repo" >&2; exit 1; fi
  step "Checkout $repo@$ref" git -C "$repo" checkout "$ref"
  if [[ "$ref" == "master" ]]; then step "Pull $repo master" git -C "$repo" pull --ff-only || true; fi
}

[[ -d "$HYPRE_SRC" ]] || { echo "Hypre repo not found: $HYPRE_SRC" >&2; exit 1; }
[[ -d "$MFEM_SRC" ]]  || { echo "MFEM repo not found:  $MFEM_SRC" >&2; exit 1; }
[[ -d "$METIS_SRC" ]] || { echo "METIS not found:     $METIS_SRC" >&2; exit 1; }

checkout_ref "$HYPRE_SRC" "$HYPRE_TAG"
checkout_ref "$MFEM_SRC" "$MFEM_TAG"

# 6) Build METIS (in-place; MFEM finds lib and headers here)
step "Build METIS" bash -c "
  set -euo pipefail
  cd '$METIS_SRC' && make distclean || true
  cd '$METIS_SRC' && rm -rf lib && make -j BUILDDIR=lib config && make -j BUILDDIR=lib
  cp -f '$METIS_SRC/lib/libmetis/libmetis.a' '$METIS_SRC/lib' || true
"

# 7) Build Hypre (autotools in-tree under hypre/src)
HYPRE_AUTOTOOLS_SRC="$HYPRE_SRC/src"
[[ -d "$HYPRE_AUTOTOOLS_SRC" ]] || { echo "Hypre src/ not found in $HYPRE_SRC" >&2; exit 1; }

step "Configure Hypre → $H_INST (in-tree)" bash -c "
  set -euo pipefail
  cd \"$HYPRE_AUTOTOOLS_SRC\"
  make distclean >/dev/null 2>&1 || true
  [[ -x ./configure ]] || autoreconf -i
  PREC_FLAG=\$( [[ \"\${PRECISION}\" == single ]] && echo --enable-single || echo '' )
  CUDA_FLAG=\$( [[ \"\${WITH_CUDA}\" == 1 ]] && echo --with-cuda || echo '' )
  GPU_FLAG=
  if [[ \"\${GPU_AWARE}\" == 0 ]]; then
    if ./configure --help 2>/dev/null | grep -q -- '--disable-gpu-aware-mpi'; then
      GPU_FLAG=--disable-gpu-aware-mpi
    else
      echo 'Note: hypre configure has no --disable-gpu-aware-mpi; relying on autodetect' >&2
    fi
  fi
  FORT_FLAG=--disable-fortran
  if [[ \"\${WITH_MPI}\" == 1 ]]; then
    env ${GPU_CC:+HYPRE_CUDA_SM=${GPU_CC}} CC=mpicc CXX=mpic++ ./configure --prefix=\"$H_INST\" \$FORT_FLAG \$PREC_FLAG \$CUDA_FLAG \$GPU_FLAG
  else
    env ${GPU_CC:+HYPRE_CUDA_SM=${GPU_CC}} CC=gcc CXX=g++ ./configure --prefix=\"$H_INST\" \$FORT_FLAG \$PREC_FLAG \$CUDA_FLAG \$GPU_FLAG
  fi
"

step "Build + install Hypre" bash -c "set -euo pipefail; cd \"$HYPRE_AUTOTOOLS_SRC\" && make -j && make install"

# Hypre metadata
{
  echo "name=hypre"
  echo "variant=$VARIANT_KEY"
  echo -n "git_rev="; git -C "$HYPRE_SRC" rev-parse --short HEAD
  echo -n "git_ref="; git -C "$HYPRE_SRC" describe --tags --exact-match 2>/dev/null || echo master
  echo -n "gpu_aware_compile_macro="; grep -n "HYPRE_USING_GPU_AWARE_MPI" "$H_INST/include/HYPRE_config.h" || true
  echo "gpu_aware_requested=$GPU_AWARE"
  echo "configured_with_cuda=$WITH_CUDA"
  echo "precision=$PRECISION"
  echo "timestamp=$(date -Is)"
} > "$H_INST/metadata.txt"

# 8) Build MFEM against installed Hypre
MFEM_PREC_FLAG=$([[ "$PRECISION" == single ]] && echo MFEM_PRECISION=single || echo '')

step "MFEM clean" bash -c "set -euo pipefail; cd '$MFEM_SRC' && make clean || true"

# Serial MFEM
step "MFEM serial" bash -c "set -euo pipefail; cd '$MFEM_SRC' && make -j serial $MFEM_PREC_FLAG"

# Parallel MFEM with METIS and OpenMP (CPU-only). Skip if building CUDA variant to avoid conflicts
if [[ "$WITH_MPI" == 1 && "${WITH_CUDA}" != 1 ]]; then
  step "MFEM parallel" bash -c "set -euo pipefail; cd '$MFEM_SRC' && make -j parallel $MFEM_PREC_FLAG MFEM_USE_METIS_5=YES METIS_DIR='$METIS_SRC' MFEM_USE_OPENMP=YES MFEM_THREAD_SAFE=YES HYPRE_DIR='$H_INST'"
fi

# CUDA builds (serial + parallel CUDA)
if [[ "$WITH_CUDA" == 1 ]]; then
  # Try to detect toolkit
  if [[ -z "${CUDA_DIR:-}" && -x "$(command -v nvcc)" ]]; then CUDA_DIR="$(dirname "$(dirname "$(command -v nvcc)")")"; fi
  step "MFEM cuda" bash -c "set -euo pipefail; cd '$MFEM_SRC' && make -j cuda $MFEM_PREC_FLAG CUDA_DIR='$CUDA_DIR' CUDA_ARCH='${CUDA_ARCH:-}'"
  if [[ "$WITH_MPI" == 1 ]]; then
    step "MFEM pcuda" bash -c "set -euo pipefail; cd '$MFEM_SRC' && make -j pcuda $MFEM_PREC_FLAG MFEM_USE_METIS_5=YES METIS_DIR='$METIS_SRC' CUDA_DIR='$CUDA_DIR' CUDA_ARCH='${CUDA_ARCH:-}' HYPRE_DIR='$H_INST'"
  fi
fi

# Optional MFEM install (if supported); ignore errors
make -C "$MFEM_SRC" install prefix="$M_INST" || true

{
  echo "name=mfem"
  echo "variant=$VARIANT_KEY"
  echo -n "git_rev="; git -C "$MFEM_SRC" rev-parse --short HEAD
  echo -n "git_ref="; git -C "$MFEM_SRC" describe --tags --exact-match 2>/dev/null || echo master
  echo "precision=$PRECISION"
  echo "with_cuda=$WITH_CUDA"
  echo "with_mpi=$WITH_MPI"
  echo "timestamp=$(date -Is)"
} > "$M_INST/metadata.txt"

printf "\nBuild complete. Variant: %s\n" "$VARIANT_KEY"
printf "HYPRE_DIR=%s\n" "$H_INST"
printf "MFEM_DIR=%s (if installed) or %s (source tree)\n" "$M_INST" "$MFEM_SRC"
