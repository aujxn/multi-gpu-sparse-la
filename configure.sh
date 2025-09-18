#!/usr/bin/env bash
set -euo pipefail

# Configure this project with CMake in a reproducible, documented way.
#
# Usage examples:
#   ./configure.sh --hypre-dir /scratch/.../deps/<variant>/hypre \
#                  --mfem-dir  /scratch/.../deps/<variant>/mfem \
#                      --type RelWithDebInfo
#
#   CUDA=ON NCCL_HOME=$HOME/opt/src/nccl ./configure.sh \
#     --hypre-dir /scratch/.../deps/mpi-cuda-.../hypre --type Release
#
# Options (flags) and environment variables:
#   --build-dir DIR     Build directory (default: build)
#   --type TYPE         CMAKE_BUILD_TYPE (Debug|Release|RelWithDebInfo) [default: RelWithDebInfo]
#   --hypre-dir DIR     REQUIRED: Hypre install prefix (must contain include/HYPRE.h and lib/libHYPRE.a)
#   --mfem-dir DIR      MFEM install/source prefix (libmfem.a and include/)
#   --spdlog-dir DIR    spdlog source dir (header-only). Default: $HOME/opt/src/spdlog
#   --cuda-arch ARCH    CMAKE_CUDA_ARCHITECTURES value (e.g. 89 for L40S). If set, CUDA=ON is implied
#   --cmake CMAKE       Path to cmake executable (default: cmake)
#
#   Environment:
#     CUDA=ON|OFF       Enable CUDA language (default: ON if found; OFF otherwise)
#     NCCL_HOME         Root containing include/nccl.h and lib/libnccl.so (optional; auto-detected under $HOME/opt/src)
#     HYPRE_DIR         Alternative to --hypre-dir
#     MFEM_DIR          Alternative to --mfem-dir
#     SPDLOG_DIR        Alternative to --spdlog-dir

usage() { sed -n '1,60p' "$0"; exit 1; }

BUILD_DIR=build
BUILD_TYPE=RelWithDebInfo
HYPRE_DIR_ARG="${HYPRE_DIR:-}"
MFEM_DIR_ARG="${MFEM_DIR:-}"
SPDLOG_DIR_ARG="${SPDLOG_DIR:-${HOME}/opt/src/spdlog}"
CMAKE_BIN=cmake
CUDA_ARCH_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --type) BUILD_TYPE="$2"; shift 2;;
    --hypre-dir) HYPRE_DIR_ARG="$2"; shift 2;;
    --mfem-dir) MFEM_DIR_ARG="$2"; shift 2;;
    --spdlog-dir) SPDLOG_DIR_ARG="$2"; shift 2;;
    --cuda-arch) CUDA_ARCH_ARG="$2"; shift 2;;
    --cmake) CMAKE_BIN="$2"; shift 2;;
    *) echo "Unknown option: $1" >&2; usage;;
  esac
done

CUDA=${CUDA:-}
if [[ -n "$CUDA_ARCH_ARG" ]]; then CUDA=ON; fi

if [[ -z "$HYPRE_DIR_ARG" ]]; then
  echo "ERROR: --hypre-dir (or HYPRE_DIR) is required." >&2
  exit 1
fi

# Basic validations
[[ -f "$HYPRE_DIR_ARG/include/HYPRE.h" ]] || { echo "Missing $HYPRE_DIR_ARG/include/HYPRE.h" >&2; exit 1; }
[[ -f "$HYPRE_DIR_ARG/lib/libHYPRE.a" ]] || { echo "Missing $HYPRE_DIR_ARG/lib/libHYPRE.a" >&2; exit 1; }

if [[ -n "$MFEM_DIR_ARG" ]]; then
  if [[ ! -f "$MFEM_DIR_ARG/libmfem.a" && ! -f "$MFEM_DIR_ARG/lib/libmfem.a" ]]; then
    echo "WARN: libmfem.a not found under $MFEM_DIR_ARG; proceeding anyway." >&2
  fi
fi

if [[ -z "${NCCL_HOME:-}" ]]; then
  # Best-effort: leave to CMake find_path/find_library under $HOME/opt/src
  echo "NCCL_HOME not set; CMake will search under $HOME/opt/src." >&2
fi

if [[ ! -d "$SPDLOG_DIR_ARG/include/spdlog" ]]; then
  echo "ERROR: spdlog headers not found under $SPDLOG_DIR_ARG/include/spdlog" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

GEN_ARGS=(
  -DHYPRE_DIR="$HYPRE_DIR_ARG"
  -DSPDLOG_DIR="$SPDLOG_DIR_ARG"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [[ -n "$MFEM_DIR_ARG" ]]; then
  GEN_ARGS+=( -DMFEM_DIR="$MFEM_DIR_ARG" )
fi

# Intentionally do not pass -DNCCL_HOME; CMakeLists reads $ENV{NCCL_HOME}.

if [[ "${CUDA:-}" == "ON" ]]; then
  if [[ -n "$CUDA_ARCH_ARG" ]]; then
    GEN_ARGS+=( -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_ARG" )
  fi
fi

echo "Configuring in $BUILD_DIR with:" 
printf '  %s\n' "${GEN_ARGS[@]}"

"$CMAKE_BIN" -S . -B "$BUILD_DIR" "${GEN_ARGS[@]}"

echo "Done. Build with: cmake --build $BUILD_DIR -j"
