#!/usr/bin/env bash
# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
set -euo pipefail

# RDT2 policy environment setup.
#
# The upstream RDT2 source tree is NOT vendored into XPolicyLab and is NOT a
# submodule (see docs/rdt2-umi-runbook.md, "git 追踪"): making it one would put a
# third level of nesting under manimux -> XPolicyLab -> RDT2, and we do not patch
# upstream, so there is nothing to commit into a fork. It is pinned by commit
# instead, the same way policy/LingBot_VLA pins lerobot.
#
# Everything below is the environment that was actually validated end to end on
# one RTX 5090 (1500 steps, batch 8, loss 0.0192 -> 0.0057). Upstream's
# requirements.txt pins neither torch nor flash-attn, which is the gap this
# script closes.
#
# Usage:
#   bash install.sh [rdt2_root]
#
# Honors: RDT2_ROOT, RDT2_GIT_COMMIT, RDT2_ENV_DIR, RDT2_TORCH_INDEX_URL,
#         RDT2_FLASH_ATTN_WHEEL, RDT2_FLASH_ATTN_VERSION, MAX_JOBS.
# Behind a censored resolver, export UV_DEFAULT_INDEX to a PyPI mirror rather
# than routing bulk downloads through a proxy.

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

# Pinned upstream. Bump deliberately; the arm-order / gripper-unit contract in
# model.py cites line numbers in this tree.
RDT2_GIT_REPO="https://github.com/thu-ml/RDT2.git"
RDT2_GIT_COMMIT="${RDT2_GIT_COMMIT:-0797b4c65e588e088d41602685e00dc2bc95852f}"

# 2.8.3.post1 is the first upstream release whose setup.py emits
# `arch=compute_120,code=sm_120`. There is no prebuilt wheel covering Blackwell,
# and rdt/train.py:134 hardcodes attn_implementation="flash_attention_2" with no
# fallback, so this is not optional on a 50-series card.
FLASH_ATTN_VERSION="${RDT2_FLASH_ATTN_VERSION:-2.8.3.post1}"

RDT2_ROOT="${1:-${RDT2_ROOT:-}}"
if [[ -z "${RDT2_ROOT}" ]]; then
    RDT2_ROOT="$(cd "${XPL_ROOT}/../.." 2>/dev/null && pwd)/RDT2"
fi
ENV_DIR="${RDT2_ENV_DIR:-${RDT2_ROOT}/.venv}"
TORCH_INDEX_URL="${RDT2_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "[RDT2][install] RDT2_ROOT   = ${RDT2_ROOT}"
echo "[RDT2][install] pinned      = ${RDT2_GIT_COMMIT}"
echo "[RDT2][install] ENV_DIR     = ${ENV_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# ---------------------------------------------------------------- upstream tree
if [[ ! -d "${RDT2_ROOT}/.git" ]]; then
    echo "[RDT2][install] cloning upstream into ${RDT2_ROOT}"
    mkdir -p "$(dirname "${RDT2_ROOT}")"
    git clone "${RDT2_GIT_REPO}" "${RDT2_ROOT}"
fi
git -C "${RDT2_ROOT}" fetch --depth 1 origin "${RDT2_GIT_COMMIT}" 2>/dev/null || true
if ! git -C "${RDT2_ROOT}" cat-file -e "${RDT2_GIT_COMMIT}^{commit}" 2>/dev/null; then
    echo "[RDT2][install] pinned commit ${RDT2_GIT_COMMIT} is not in ${RDT2_ROOT}." >&2
    echo "[RDT2][install] Fetch it manually, or override RDT2_GIT_COMMIT." >&2
    exit 1
fi
# Refuse to silently discard local upstream edits: we deliberately keep this tree
# unmodified, so a dirty tree means someone patched what the adapter assumes.
if [[ -n "$(git -C "${RDT2_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    echo "[RDT2][install] upstream tree has local modifications to tracked files." >&2
    echo "[RDT2][install] The adapter assumes stock upstream. Resolve, then re-run." >&2
    git -C "${RDT2_ROOT}" status --short --untracked-files=no >&2
    exit 1
fi
# Only move HEAD when it is actually wrong: a tree already sitting on the pinned
# commit via a branch should not be gratuitously detached.
if [[ "$(git -C "${RDT2_ROOT}" rev-parse HEAD)" != "${RDT2_GIT_COMMIT}" ]]; then
    git -C "${RDT2_ROOT}" checkout --quiet "${RDT2_GIT_COMMIT}"
fi
echo "[RDT2][install] upstream at $(git -C "${RDT2_ROOT}" rev-parse --short HEAD)"

# ------------------------------------------------------------------------ venv
uv venv --python 3.10 "${ENV_DIR}"
MODEL_PYTHON="${ENV_DIR}/bin/python"

# torch first: flash-attn compiles against it, and the cu126 wheel PyPI serves by
# default has no sm_120 kernels.
uv pip install --python "${MODEL_PYTHON}" \
    torch==2.7.1 torchvision==0.22.1 --index-url "${TORCH_INDEX_URL}"

# vllm is dropped on purpose: it is imported lazily by deploy/vllm_utils.py and
# deploy/inference_real_vq.py only, i.e. the VQ inference path. Nothing under
# rdt/ touches it, and resolving it drags in a conflicting torch.
grep -v '^vllm' "${RDT2_ROOT}/requirements.txt" \
    | uv pip install --python "${MODEL_PYTHON}" -r -

# ------------------------------------------------------------------ flash-attn
if "${MODEL_PYTHON}" -c "import flash_attn" 2>/dev/null; then
    echo "[RDT2][install] flash-attn already present, skipping"
elif [[ -n "${RDT2_FLASH_ATTN_WHEEL:-}" && -f "${RDT2_FLASH_ATTN_WHEEL}" ]]; then
    uv pip install --python "${MODEL_PYTHON}" --no-deps "${RDT2_FLASH_ATTN_WHEEL}"
else
    ARCH="$("${MODEL_PYTHON}" -c \
        'import torch;m,n=torch.cuda.get_device_capability();print(f"{m}{n}")' 2>/dev/null || true)"
    : "${ARCH:=120}"
    echo "[RDT2][install] building flash-attn ${FLASH_ATTN_VERSION} for sm_${ARCH} (30-80 min)"

    # conda-forge's cuda-toolkit puts headers under targets/<triple>/include, but
    # torch's cpp_extension only adds $CUDA_HOME/include. Build a shadow root with
    # the layout it expects rather than requiring a system CUDA install.
    CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    if [[ ! -f "${CUDA_HOME}/include/cuda_runtime.h" \
          && -d "${CUDA_HOME}/targets/x86_64-linux/include" ]]; then
        SHADOW="${ENV_DIR}/.cuda-home"
        mkdir -p "${SHADOW}"
        ln -sfn "${CUDA_HOME}/bin" "${SHADOW}/bin"
        ln -sfn "${CUDA_HOME}/targets/x86_64-linux/include" "${SHADOW}/include"
        ln -sfn "${CUDA_HOME}/targets/x86_64-linux/lib" "${SHADOW}/lib64"
        CUDA_HOME="${SHADOW}"
        echo "[RDT2][install] using shadow CUDA_HOME=${CUDA_HOME}"
    fi

    # Each nvcc job on these template-heavy kernels peaks around 4GB.
    JOBS="${MAX_JOBS:-4}"
    CUDA_HOME="${CUDA_HOME}" \
    PATH="${CUDA_HOME}/bin:${PATH}" \
    FLASH_ATTN_CUDA_ARCHS="${ARCH}" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    MAX_JOBS="${JOBS}" \
    uv pip install --python "${MODEL_PYTHON}" --no-build-isolation \
        "flash-attn==${FLASH_ATTN_VERSION}"
fi

# ------------------------------------------------------------------ XPolicyLab
uv pip install --python "${MODEL_PYTHON}" -e "${XPL_ROOT}"

"${MODEL_PYTHON}" - <<'PY'
import torch, flash_attn, transformers, webdataset, XPolicyLab  # noqa: F401
archs = torch.cuda.get_arch_list()
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
print(f"flash_attn {flash_attn.__version__}  transformers {transformers.__version__}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    want = f"sm_{major}{minor}"
    assert want in archs, f"{want} missing from torch arch list {archs}"
    print(f"{torch.cuda.get_device_name(0)} -> {want} OK")
print("RDT2 env ok")
PY

echo "[RDT2][install] done."
echo "[RDT2][install] export RDT2_ROOT=${RDT2_ROOT} (or set rdt2_root in deploy.yml)"
