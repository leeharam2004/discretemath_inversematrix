import numpy as np
import mat_tools

EPS = 1e-12  # 피벗이 이 값보다 작으면 0으로 간주 (부동소수점 오차 대비)


def inv_mat(mat: np.ndarray) -> np.ndarray:
    if not mat_tools.mat_squarecheck(mat):
        raise ValueError("입력된 행렬이 정방행렬이 아닙니다.")
    mat_dim = mat.shape[0]

    aug = np.hstack((mat.astype(float), np.eye(mat_dim)))

    for col in range(mat_dim):

        pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot_row, col]) < EPS:
            raise ValueError("피벗이 0이므로 역행렬이 존재하지 않습니다.")

        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]

        aug[col] = aug[col] / aug[col, col]

        for row in range(mat_dim):
            if row != col and aug[row, col] != 0:
                aug[row] = aug[row] - aug[row, col] * aug[col]

    return aug[:, mat_dim:]