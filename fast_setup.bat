@echo off
echo [*] Protein Physics Engine - 환경 구축을 시작합니다.
echo [1/3] 필수 라이브러리 설치 중...
pip install pybind11 PyQt6 PyQt6-WebEngine biopython numpy

echo [2/3] C++ 물리 엔진 빌드 중 (MSVC 컴파일러 필요)...
python setup.py build_ext --inplace

echo [3/3] 프로그램 실행...
python gui_main.py
pause