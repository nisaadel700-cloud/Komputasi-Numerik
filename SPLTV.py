# Cell 1: Import library
import numpy as np # NumPy menyediakan array, perkalian matriks (@), dot product, dan argmax

# cell 2: Fungsi untuk menampilkan matriks augmented
def tampil_matriks(M, judul):                      # M = matriks augmented 3x4, judul = penanda langkah
    print(judul)                                   # cetak judul agar jelas langkah mana yang sedang ditampilkan
    for baris in M:                                # telusuri matriks baris demi baris
        koefisien = "  ".join(f"{v:9.4f}" for v in baris[:-1])
        # baris[:-1] = tiga kolom koefisien; format 9.4f = lebar 9 karakter, 4 angka desimal agar kolom sejajar
        print(f"  [ {koefisien}  |  {baris[-1]:9.4f} ]")
        # baris[-1] = kolom konstanta b; garis tegak | memisahkan koefisien dan konstanta
    print()                                        # baris kosong sebagai jarak antarlangkah

# Cell 3: Fungsi utama eliminasi Gauss dengan pivoting parsial
def eliminasi_gauss(A, b, tol=1e-12, tampil=True):  # tol = batas pivot dianggap nol, tampil = tampilkan langkah atau tidak
    A = np.array(A, dtype=float)                    # ubah A menjadi array bertipe desimal agar pembagian tidak terpotong jadi bilangan bulat
    b = np.array(b, dtype=float)                    # ubah b dengan alasan yang sama
    if A.shape != (3, 3) or b.shape != (3,):        # validasi: A harus 3x3 dan b harus 3 elemen
        raise ValueError("Ukuran harus A 3x3 dan b 3x1.")  # hentikan program bila ukuran salah
    n = 3                                           # jumlah variabel (dan jumlah persamaan)
    M = np.hstack((A, b.reshape(n, 1)))             # gabungkan A dan b menjadi matriks augmented 3x4
    if tampil:                                      # tampilkan hanya bila diminta
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
