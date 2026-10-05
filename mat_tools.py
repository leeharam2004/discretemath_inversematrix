import numpy as np

def mat_input() -> tuple[np.ndarray, int]:
    mat_dim = int(input("행렬의 차수를 입력하세요. 정방행렬만 가능합니다."))
    mat = np.empty((mat_dim, mat_dim))
    for i in range(mat_dim):
        while True:
            try:
                mat[i] = np.array(list(map(float, input(f"{i}행을 띄어쓰기로 구분하여 입력하세요.").split())))
                break
            except ValueError:
                print(f"잘못 입력하였습니다. {mat_dim} 개의 원소를 띄어쓰기로 구분하여 입력하세요.")
    return mat, mat_dim

def mat_squarecheck(mat: np.ndarray) -> bool:
    return (mat.ndim == 2 and mat.shape[0] == mat.shape[1])