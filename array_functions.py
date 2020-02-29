# Array functions

def to_1d(arr3d):
    arr1d = []
    for i in range(len(arr3d)):
        for j in range(len(arr3d[i])):
            for k in range(len(arr3d[i][j])):
                arr1d.append(arr3d[i][j][k])
    return arr1d


def ijk(arr_3d, p):  # предполагается матрица (т.е массив, у которого длина подмассивов на всех уровнях одинакова)
    i = p // len(arr_3d[0][0]) // len(arr_3d[0])
    j = p // len(arr_3d[0][0]) % len(arr_3d[0])
    k = p % len(arr_3d[0][0])
    return i, j, k


def a3_ijk(arr3d, p):
    i, j, k = ijk(arr3d, p)
    return arr3d[i][j][k]


def ijkd(arr_4d, p):  # предполагается матрица (т.е массив, у которого длина подмассивов на всех уровнях одинакова)
    i = p // len(arr_4d[0][0][0]) // len(arr_4d[0][0]) // len(arr_4d[0])
    j = p // len(arr_4d[0][0][0]) // len(arr_4d[0][0]) % len(arr_4d[0])
    k = p // len(arr_4d[0][0][0]) % len(arr_4d[0][0])
    d = p % len(arr_4d[0][0][0])
    return i, j, k, d


def a4_ijk(arr4d, p):
    i, j, k, d = ijkd(arr4d, p)
    return arr4d[i][j][k][d]
