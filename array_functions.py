# Array functions

# TODO: make the 1 function for every dimension, it should return tuple (n, list_1d)

def list2d_to_1d(arr2d):
    arr1d = []
    for i in range(len(arr2d)):
        for j in range(len(arr2d[i])):
            arr1d.append(arr2d[i][j])
    return arr1d


def list3d_to_1d(arr3d):
    arr1d = []
    for i in range(len(arr3d)):
        for j in range(len(arr3d[i])):
            for k in range(len(arr3d[i][j])):
                arr1d.append(arr3d[i][j][k])
    return arr1d


def list4d_to_1d(arr4d):
    arr1d = []
    for i in range(len(arr4d)):
        for j in range(len(arr4d[i])):
            for k in range(len(arr4d[i][j])):
                for d in range(len(arr4d[i][j][k])):
                    arr1d.append(arr4d[i][j][k][d])
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


def p3(arr3d, i, j, k):
    return len(arr3d) * len(arr3d[0]) * i + len(arr3d) * j + k


def p4(arr4d, i, j, k, d):
    return len(arr4d) * len(arr4d[0]) * len(arr4d[0][0]) * i + len(arr4d) * len(arr4d[0]) * j + len(arr4d) * k + d
