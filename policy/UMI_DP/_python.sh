# Sourced by standard entry points. Argument is a venv path or conda env name.
umi_select_python() {
    if [[ -x "$1/bin/python" ]]; then
        UMI_PYTHON="$(cd "$1" && pwd)/bin/python"
    elif [[ -x "$1" && -f "$1" ]]; then
        UMI_PYTHON="$1"
    else
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$1"
        UMI_PYTHON="$(command -v python)"
    fi
}
