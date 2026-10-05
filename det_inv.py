import numpy as np
import mat_tools

def submat(a, b, mat: np.ndarray) -> np.ndarray: # 행렬번호(a, b) 0부터 카운팅
    mat = np.delete(mat, a, 0)
    mat = np.delete(mat, b, 1)
    return mat

def det(mat: np.ndarray):
    if not mat_tools.mat_squarecheck(mat):
        raise ValueError("입력된 행렬이 정방행렬이 아닙니다.")
    mat_dim = mat.shape[0]
    
    if mat_dim == 1:
        return mat[0][0]
    elif mat_dim == 2:
        return mat[0][0] * mat[1][1] - mat[0][1]*mat[1][0]
    else:
        out = 0
        for i, val in enumerate(mat[0]):
            out += cofactor(0, i, mat) * val
        return out


def cofactor(a, b, mat: np.ndarray) -> float: # 행렬번호(a, b) 0부터 카운팅
    out = det(submat(a, b, mat))
    if (a + b) % 2 != 0: out *= -1
    return out

def cofactor_mat(mat: np.ndarray) -> np.ndarray:
    if not mat_tools.mat_squarecheck(mat):
            raise ValueError("입력된 행렬이 정방행렬이 아닙니다.")
    mat_dim = mat.shape[0]
    out = np.ndarray((mat_dim, mat_dim))

    for i in range(mat_dim):
        for j in range(mat_dim):
            out[i][j] = cofactor(i, j, mat)
    return out

def inv_mat(mat):
    if not mat_tools.mat_squarecheck(mat):
            raise ValueError("입력된 행렬이 정방행렬이 아닙니다.")
    mat_det = det(mat)

    if mat_det == 0:
         raise ValueError("행렬식이 0이므로 역행렬이 존재하지 않습니다.")
    
    adj = cofactor_mat(mat).transpose()
    out = adj / mat_det
    return out