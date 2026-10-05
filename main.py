import numpy as np
import mat_tools
import det_inv
import gauss_jordan_inv

np.set_printoptions(precision=6, suppress=True)
TOL = 1e-9  # 두 결과가 같다고 볼 허용 오차


def run_once():
    # 1. 행렬 입력
    mat, n = mat_tools.mat_input()
    print("\n입력된 행렬 A:")
    print(mat)

    # 행렬식을 통한 역행렬
    det_result = None
    try:
        det_result = det_inv.inv_mat(mat)
        print("\n[행렬식 방법] 역행렬:")
        print(det_result)
    except ValueError as e:
        print(f"\n[행렬식 방법] 오류: {e}")

    # 가우스-조던 역행렬
    gj_result = None
    try:
        gj_result = gauss_jordan_inv.inv_mat(mat)
        print("\n[가우스-조던 방법] 역행렬:")
        print(gj_result)
    except ValueError as e:
        print(f"\n[가우스-조던 방법] 오류: {e}")

    # 결과 비교
    print("\n[결과 비교]: ")
    if det_result is None and gj_result is None:
        print("역행렬이 존재하지 않습니다. (일치)")
        return
    if det_result is None or gj_result is None:
        print("두 방법의 판정이 서로 다릅니다. (불일치)")
        return

    if np.allclose(det_result, gj_result, atol=TOL):
        print("두 방법으로 계산한 역행렬이 동일합니다.")
    else:
        print("두 방법으로 계산한 역행렬이 다릅니다.")
        return

    # 추가 기능: 검산 A x A^-1 = I
    print("\n[검산] A x A^-1:")
    print(mat @ gj_result)
    if np.allclose(mat @ gj_result, np.eye(n), atol=TOL):
        print("검산 성공")
    else:
        print("검산 실패")

    return


def main():
    while True:
        run_once()
        again = input("\n다시 실행하시겠습니까? (y/n): ").strip().lower()
        if again != "y":
            break
        print("\n" + "=" * 40)


if __name__ == "__main__":
    main()