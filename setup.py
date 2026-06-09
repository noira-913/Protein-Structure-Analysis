"""
pip install -e .          # dev mode (CPU only)
pip install .             # normal install (CPU only)
python setup.py build_ext --inplace   # builds both CPU and GPU if CUDA found

Auto-detection:
  - OpenMP   -> multi-core CPU parallelism
  - CUDA     -> GPU-accelerated engine (protein_physics_cuda)
"""

import sys
import os
import glob
import shutil
import sysconfig
import subprocess
import tempfile
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


# ── OpenMP detection ─────────────────────────────────────────────
def has_openmp() -> bool:
    test_src = "#include <omp.h>\nint main(){return omp_get_max_threads();}\n"
    flag = "/openmp" if sys.platform == "win32" else "-fopenmp"
    try:
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
            f.write(test_src)
            fname = f.name
        if sys.platform == "win32":
            cl = shutil.which("cl") or _find_cl_exe_path()
            if not cl:
                return False
            cmd = [cl, flag, fname, "/Fe" + os.devnull]
        else:
            cmd = ["c++", flag, fname, "-o", os.devnull]
        ret = subprocess.run(cmd, capture_output=True, timeout=10)
        return ret.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


def _find_cl_exe_path():
    """Return the full path to cl.exe, or None."""
    vswhere = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        r"Microsoft Visual Studio\Installer\vswhere.exe",
    )
    if not os.path.exists(vswhere):
        return None
    try:
        result = subprocess.run(
            [vswhere, "-latest", "-products", "*",
             "-find", r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        return lines[-1] if lines else None
    except Exception:
        return None


# ── CUDA detection ───────────────────────────────────────────────
def find_cuda():
    """Return (cuda_home, nvcc_path) or (None, None)."""
    # 1. Environment variables set by the CUDA installer
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")

    # 2. Standard Windows install locations
    if not cuda_home:
        candidates = glob.glob(
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
        if candidates:
            cuda_home = sorted(candidates)[-1]

    # 3. nvcc on PATH -> infer cuda_home
    if not cuda_home:
        nvcc_on_path = shutil.which("nvcc")
        if nvcc_on_path:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_on_path))

    if not cuda_home or not os.path.isdir(cuda_home):
        return None, None

    nvcc_name = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    nvcc = os.path.join(cuda_home, "bin", nvcc_name)
    return (cuda_home, nvcc) if os.path.exists(nvcc) else (None, None)


# ── MSVC cl.exe detection (needed by nvcc as host compiler) ──────
def find_cl_exe():
    """Return the directory containing cl.exe, or None if not found."""
    if shutil.which("cl"):
        return None  # already on PATH, no action needed
    path = _find_cl_exe_path()
    return os.path.dirname(path) if path else None


# ── Build CUDA extension via nvcc subprocess ─────────────────────
def build_cuda_extension(cuda_home, nvcc):
    """
    Compile protein_physics_cuda using nvcc directly (bypasses setuptools).
    Returns True on success.
    """
    import pybind11

    py_include  = sysconfig.get_path("include")
    pb11_include = pybind11.get_include()
    cuda_include = os.path.join(cuda_home, "include")

    # Output filename must match Python's extension suffix
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    output = f"protein_physics_cuda{ext_suffix}"

    is_win = sys.platform == "win32"
    cuda_lib = os.path.join(cuda_home, "lib", "x64" if is_win else "../lib64")
    host_opts = "/MD /O2 /EHsc /D_USE_MATH_DEFINES" if is_win else "-fPIC"
    compiler_flag = "--compiler-options" if is_win else "-Xcompiler"

    cmd = [
        nvcc, "-O2",
        compiler_flag, host_opts,
        f"-I{py_include}", f"-I{pb11_include}", f"-I{cuda_include}",
        "--shared",
        "src/physics_engine_cuda.cu",
        "-o", output,
        f"-L{cuda_lib}", "-lcudart",
    ]
    if is_win:
        py_ver = f"python{sys.version_info.major}{sys.version_info.minor}"
        cmd.append(os.path.join(sys.base_prefix, "libs", f"{py_ver}.lib"))

    # Ensure cl.exe is on PATH for nvcc's host compiler
    env = os.environ.copy()
    cl_dir = find_cl_exe()
    if cl_dir:
        env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")
        print(f"  cl.exe found at: {cl_dir}")

    print(f"Building CUDA extension: {output}")
    ret = subprocess.run(cmd, capture_output=False, env=env)
    if ret.returncode != 0:
        print("CUDA build failed — GPU backend will not be available.")
        return False
    print(f"CUDA extension built: {output}")
    return True


# ── Compile flags ────────────────────────────────────────────────
if sys.platform == "win32":
    extra_compile = ["/O2", "/arch:AVX2"]
    extra_link    = []
    omp_flag      = "/openmp"
else:
    extra_compile = ["-O3", "-march=native", "-ffast-math"]
    extra_link    = []
    omp_flag      = "-fopenmp"

openmp_available = has_openmp()
if openmp_available:
    extra_compile.append(omp_flag)
    extra_link.append(omp_flag)
    print("OpenMP detected -> multi-core CPU build enabled")
else:
    print("OpenMP undetected -> single-thread CPU build")

cuda_home, nvcc = find_cuda()
if cuda_home:
    print(f"CUDA detected at: {cuda_home}")
else:
    print("CUDA not found -> GPU backend will not be built")


# ── CPU extension ────────────────────────────────────────────────
cpu_ext = Pybind11Extension(
    "protein_physics",
    sources=["src/physics_engine.cpp"],
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
)


# ── Custom build_ext: builds CPU via setuptools, CUDA via nvcc ───
class CustomBuildExt(build_ext):
    def run(self):
        super().run()
        if cuda_home and nvcc:
            build_cuda_extension(cuda_home, nvcc)


setup(
    name="protein_physics",
    version="0.2.0",
    author="",
    description="Implicit-solvent protein physics engine (CPU + optional GPU)",
    ext_modules=[cpu_ext],
    cmdclass={"build_ext": CustomBuildExt},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
)
