"""Headless EGL / MuJoCo bootstrap before robosuite or cpgen_envs imports."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import sys
from typing import List, Optional

_PRELOADED = False


def ensure_eval_gl_env() -> None:
    """Apply eval-safe GL defaults and validate EGL vendor paths."""
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", ""),
    )
    _egl_vendor = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "")
    if _egl_vendor and not os.path.isfile(_egl_vendor):
        print(
            f"[eval] WARN: unset missing __EGL_VENDOR_LIBRARY_FILENAMES={_egl_vendor}"
        )
        os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)

    workspace = _workspace_root()
    if workspace:
        roots = _list_staged_egl_roots(workspace, _driver_version())
        if roots:
            _apply_staged_egl(roots[0], workspace)
            _preload_egl_library(os.path.join(roots[0], "lib"))


def _workspace_root() -> str:
    return os.environ.get("workspace") or os.environ.get("WORKSPACE") or ""


def _find_generic_glvnd_backup(workspace: str) -> Optional[str]:
    for cand in sorted(
        glob.glob(os.path.join(workspace, ".nvidia_egl_*", "lib", ".generic_glvnd_backup_*"))
    ):
        if os.path.isfile(os.path.join(cand, "libEGL.so.1")):
            return cand
    return None


def _driver_version() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .split("\n", 1)[0]
            .strip()
        )
    except Exception:
        return ""


def _list_staged_egl_roots(workspace: str, drv: str) -> List[str]:
    roots: List[str] = []
    if drv:
        exact = os.path.join(workspace, f".nvidia_egl_{drv}")
        if os.path.isdir(os.path.join(exact, "lib")):
            roots.append(exact)
        major = drv.split(".", 1)[0]
        for cand in sorted(glob.glob(os.path.join(workspace, f".nvidia_egl_{major}.*"))):
            if os.path.isdir(os.path.join(cand, "lib")) and cand not in roots:
                roots.append(cand)
    for cand in sorted(glob.glob(os.path.join(workspace, ".nvidia_egl_*"))):
        if os.path.isdir(os.path.join(cand, "lib")) and cand not in roots:
            roots.append(cand)
    return roots


def _apply_staged_egl(root: str, workspace: str) -> None:
    lib_dir = os.path.join(root, "lib")
    vendor = os.path.join(root, "share", "glvnd", "egl_vendor.d", "10_nvidia.json")
    generic = None
    for cand in glob.glob(os.path.join(lib_dir, ".generic_glvnd_backup_*")):
        if os.path.isfile(os.path.join(cand, "libEGL.so.1")):
            generic = cand
            break
    if generic is None and workspace:
        generic = _find_generic_glvnd_backup(workspace)
    parts = []
    if generic:
        parts.append(generic)
    parts.append(lib_dir)
    parts.append(os.environ.get("LD_LIBRARY_PATH", ""))
    os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in parts if p)
    if os.path.isfile(vendor):
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor
    else:
        os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)


def _preload_egl_library(staged_lib_dir: Optional[str] = None) -> None:
    global _PRELOADED
    if _PRELOADED:
        return
    workspace = _workspace_root()
    libs: List[str] = []
    generic = _find_generic_glvnd_backup(workspace) if workspace else None
    if generic:
        for name in ("libEGL.so.1", "libOpenGL.so.0", "libGLdispatch.so.0"):
            path = os.path.join(generic, name)
            if os.path.isfile(path):
                libs.append(path)
    if staged_lib_dir:
        for pattern in ("libEGL_nvidia.so.0", "libEGL_nvidia.so.*"):
            libs.extend(sorted(glob.glob(os.path.join(staged_lib_dir, pattern))))
    for lib_name in ("EGL", "OpenGL"):
        path = ctypes.util.find_library(lib_name)
        if path:
            libs.append(path)
    seen = set()
    for path in libs:
        if path in seen:
            continue
        seen.add(path)
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    _PRELOADED = True


def _clear_gl_import_cache() -> None:
    for name in list(sys.modules):
        if name == "OpenGL" or name.startswith(
            ("OpenGL.", "mujoco", "robosuite", "cpgen_envs")
        ):
            del sys.modules[name]
    global _PRELOADED
    _PRELOADED = False


def import_robosuite(force: bool = False):
    """Import robosuite after EGL bootstrap; retry with staged EGL on failure."""
    ensure_eval_gl_env()
    _preload_egl_library()

    if not force:
        cached = sys.modules.get("robosuite")
        if cached is not None:
            return cached

    workspace = _workspace_root()
    drv = _driver_version()
    staged_roots = _list_staged_egl_roots(workspace, drv) if workspace else []
    attempts: List[Optional[str]] = [None]
    attempts.extend(staged_roots)

    last_err: Optional[BaseException] = None
    for idx, root in enumerate(attempts):
        if idx > 0 and root is not None:
            print(
                f"[eval] robosuite import failed ({last_err!r}); "
                f"retry with staged EGL {root}"
            )
            _clear_gl_import_cache()
            ensure_eval_gl_env()
            _apply_staged_egl(root, workspace)
            _preload_egl_library(os.path.join(root, "lib"))
        try:
            import robosuite as rs

            return rs
        except Exception as e:
            last_err = e
            if idx == 0 and staged_roots:
                # System ICD may be present but unusable; prefer staged vendor files.
                os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    assert last_err is not None
    raise last_err
