import protein_physics
import numpy as np


class PhysicsBridge:
    def __init__(self):
        self.engine = protein_physics.PhysicsEngine()

    def run_safe_calculation(
        self,
        coordinates,
        charges,
        water_flags,
        radii=None,
        epsilons=None,
    ):
        """
        C++로 데이터를 넘기기 전 데이터 무결성을 검사하는 안전 장치.

        Parameters
        ----------
        coordinates : array-like, shape (N, 3)
        charges     : array-like, shape (N,)
        water_flags : array-like, shape (N,)  bool
        radii       : array-like, shape (N,)  optional — 기본값 1.9 Å
        epsilons    : array-like, shape (N,)  optional — 기본값 0.1 kcal/mol
        """
        try:
            n = len(coordinates)

            # 1. 길이 일치 확인
            if not (n == len(charges) == len(water_flags)):
                raise ValueError("데이터 배열의 길이가 일치하지 않습니다.")

            # 기본값 채우기
            if radii is None:
                radii = [1.9] * n
            if epsilons is None:
                epsilons = [0.1] * n

            coords_arr = np.asarray(coordinates, dtype=float)

            # 2. NaN / Inf 검사 — 문제 원자를 건너뛰되 경고 출력
            particles = []
            skipped = 0
            for i in range(n):
                row = coords_arr[i]
                if np.any(~np.isfinite(row)):
                    print(f"  [!] 원자 {i}: 좌표에 NaN/Inf 포함 → 건너뜀")
                    skipped += 1
                    continue

                # ▶ 수정: Particle(x, y, z, charge, radius, epsilon, is_water)
                p = protein_physics.Particle(
                    float(row[0]), float(row[1]), float(row[2]),
                    float(charges[i]),
                    float(radii[i]),
                    float(epsilons[i]),
                    bool(water_flags[i]),
                )
                particles.append(p)

            if skipped:
                print(f"  [!] 총 {skipped}개 원자가 제외됐습니다.")

            if not particles:
                raise ValueError("유효한 파티클이 없습니다.")

            # 3. C++ 엔진 호출 — ▶ 수정: 실제 존재하는 메서드명 사용
            energy = self.engine.calculate_potential(particles)
            return energy

        except Exception as e:
            print(f"  [!] 물리 엔진 연동 중 오류 발생: {e}")
            return None


# ── 테스트 실행 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bridge = PhysicsBridge()

    # 정상 케이스
    test_coords  = [[0, 0, 0], [5, 0, 0], [10, 0, 0]]
    test_charges = [1.0, -1.0, 0.5]
    test_waters  = [False, False, True]

    result = bridge.run_safe_calculation(test_coords, test_charges, test_waters)
    if result is not None:
        print(f"[*] 연동 테스트 성공! 계산된 에너지: {result:.4f} kcal/mol")

    # NaN 포함 케이스 (방어 코드 확인)
    print("\n[NaN 테스트]")
    bad_coords = [[0, 0, 0], [float("nan"), 1, 1], [2, 2, 2]]
    result2 = bridge.run_safe_calculation(bad_coords, test_charges, test_waters)
    if result2 is not None:
        print(f"[*] NaN 처리 후 에너지: {result2:.4f} kcal/mol")