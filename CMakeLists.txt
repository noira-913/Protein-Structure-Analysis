cmake_minimum_required(VERSION 3.18)

# Enable CUDA only if the toolkit is present
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  project(protein_physics_project LANGUAGES CXX CUDA)
  set(CUDA_AVAILABLE TRUE)
else()
  project(protein_physics_project LANGUAGES CXX)
  set(CUDA_AVAILABLE FALSE)
  message(STATUS "CUDA not found -> GPU target skipped")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 릴리즈 빌드 최적화 ────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /arch:AVX2)
else()
  add_compile_options(-O3 -march=native -ffast-math)
endif()

# ── pybind11 ─────────────────────────────────
include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.1 # 또는 최신 버전
)

FetchContent_MakeAvailable(pybind11)

# ── OpenMP (있으면 자동 활성화) ──────────────
find_package(OpenMP)

# ── 타겟 1: 핵심 물리 엔진 ──────────────────
pybind11_add_module(protein_physics src/physics_engine.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(protein_physics PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled (multi-core build)")
else()
  message(STATUS "OpenMP undetected -> single-thread build")
endif()

# ── Target 2: GPU physics engine (optional) ─────────────────────
if(CUDA_AVAILABLE)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  pybind11_add_module(protein_physics_cuda src/physics_engine_cuda.cu)

  # Pass CUDA arch for the installed GPU (sm_75 = Turing, sm_86 = Ampere, sm_89 = Ada)
  # Falls back to a broad virtual arch if not set
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
  endif()

  set_target_properties(protein_physics_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
  )

  if(OpenMP_CXX_FOUND)
    target_link_libraries(protein_physics_cuda PRIVATE OpenMP::OpenMP_CXX)
  endif()

  target_link_libraries(protein_physics_cuda PRIVATE CUDA::cudart)
  message(STATUS "CUDA enabled -> protein_physics_cuda target added")
endif()