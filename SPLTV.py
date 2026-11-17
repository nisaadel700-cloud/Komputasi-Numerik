import numpy as np

def tampil_matriks(M, judul):
    print(judul)
    for baris in M:
        koefisien = "  ".join(f"{v:9.4f}" for v in baris[:-1])
        print(f"  [ {koefisien}  |  {baris[-1]:9.4f} ]")
    print()

def eliminasi_gauss(A, b, tol=1e-12, tampil=True):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if A.shape != (3, 3) or b.shape != (3,):
        raise ValueError("Ukuran harus A 3x3 dan b 3x1.")
    n = 3
    M = np.hstack((A, b.reshape(n, 1)))
    if tampil:
        tampil_matriks(M, "Matriks augmented awal:")

    for k in range(n - 1):
        p = k + np.argmax(np.abs(M[k:, k]))
        if abs(M[p, k]) < tol:
            raise ValueError("Pivot mendekati nol: tidak ada solusi tunggal.")
        if p != k:
            M[[k, p]] = M[[p, k]]
            if tampil:
                tampil_matriks(M, f"Tukar baris {k+1} dengan baris {p+1}:")
        for i in range(k + 1, n):
            f = M[i, k] / M[k, k]
            M[i, k:] = M[i, k:] - f * M[k, k:]
            if tampil:
                tampil_matriks(M, f"B{i+1} <- B{i+1} - ({f:.4f}) x B{k+1}:")

    if abs(M[n-1, n-1]) < tol:
        raise ValueError("Pivot terakhir nol: tidak ada solusi tunggal.")

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    return x

A = [[2, 1, -1],
     [-3, -1, 2],
     [-2, 1, 2]]
b = [8, -11, -3]

x = eliminasi_gauss(A, b)
print("Solusi sistem:")
for i, nilai in enumerate(x, start=1):
    print(f"x{i} = {nilai:.6f}")

A_np = np.array(A, dtype=float)
b_np = np.array(b, dtype=float)

residual = A_np @ x - b_np
print("Residual (A.x - b):", residual)
print("Residual maksimum  :", np.max(np.abs(residual)))

x_numpy = np.linalg.solve(A_np, b_np)
print("Solusi NumPy       :", x_numpy)
print("Sama dengan NumPy? :", np.allclose(x, x_numpy))

A_gagal = [[1, 1, 1],
           [2, 2, 2],
           [1, -1, 1]]
b_gagal = [6, 12, 2]

try:
    eliminasi_gauss(A_gagal, b_gagal, tampil=False)
except ValueError as pesan:
    print("Peringatan:", pesan)