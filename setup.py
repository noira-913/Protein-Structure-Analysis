"""
pip install -e .          # 개발 모드
pip install .             # 일반 설치

환경별 자동 감지:
  - OpenMP 있으면 멀티코어 병렬화
  - 없으면 단일 스레드 폴백 (LG 그램 등)
"""

import sys
import os
import subprocess
import tempfile
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext


def has_openmp() -> bool:
    #OpenMP 지원 여부를 컴파일 테스트로 확인
    test_src = "#include <omp.h>\nint main(){return omp_get_max_threads();}\n"
    flag = "/openmp" if sys.platform == "win32" else "-fopenmp"
    try:
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
            f.write(test_src)
            fname = f.name
        ret = subprocess.run(
            ["c++", flag, fname, "-o", os.devnull],
            capture_output=True, timeout=10
        )
        return ret.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


# ── 컴파일 플래그 ────────────────────────────
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
    print("OpenMP detected -> multi-core  build enabled")
else:
    print("OpenMP undetected -> single-thread build enabled")


ext = Pybind11Extension(
    "protein_physics",
    sources=["physics_engine.cpp"],
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
)

setup(
    name="protein_physics",
    version="0.2.0",
    author="",
    description="Implicit-solvent protein physics engine with OpenMP acceleration",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
)
