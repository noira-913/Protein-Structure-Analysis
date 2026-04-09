cmake_minimum_required(VERSION 3.18)
project(protein_physics_project LANGUAGES CXX)

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

# 이후에는 평소처럼 사용
# pybind11_add_module(protein_physics physics_engine.cpp)

# ── OpenMP (있으면 자동 활성화) ──────────────
find_package(OpenMP)

# ── 타겟 1: 핵심 물리 엔진 ──────────────────
# ▶ 수정: main.cpp 제거 — main.cpp는 'example' 모듈로 심볼 충돌 발생
pybind11_add_module(protein_physics physics_engine.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(protein_physics PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled (multi-core build)")
else()
  message(STATUS "OpenMP undetected -> single-thread build")
endif()

# ── 타겟 2: pybind11 예제 (별도 모듈) ────────
pybind11_add_module(example main.cpp)
message(STATUS "example module (main.cpp) separate build")