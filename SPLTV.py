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

    # ---------- Eliminasi maju ----------
    for k in range(n - 1):                          # k = 0 dan 1 (kolom pivot pertama dan kedua, indeks Python mulai dari 0)
        p = k + np.argmax(np.abs(M[k:, k]))         # cari baris dengan nilai mutlak terbesar pada kolom k, mulai dari baris k ke bawah
        if abs(M[p, k]) < tol:                      # pivot terbesar pun hampir nol berarti kolom ini tidak punya pivot
            raise ValueError("Pivot mendekati nol: tidak ada solusi tunggal.")
        if p != k:                                  # tukar baris hanya jika pivot terbesar bukan di baris k
            M[[k, p]] = M[[p, k]]                   # tukar baris k dan baris p sekaligus
            if tampil:
                tampil_matriks(M, f"Tukar baris {k+1} dengan baris {p+1}:")
        for i in range(k + 1, n):                   # semua baris di bawah pivot akan dinolkan pada kolom k
            f = M[i, k] / M[k, k]                   # faktor pengali = elemen yang dinolkan dibagi pivot
            M[i, k:] = M[i, k:] - f * M[k, k:]      # kurangi baris i dengan f kali baris pivot (dari kolom k sampai kolom b)
            if tampil:
                tampil_matriks(M, f"B{i+1} <- B{i+1} - ({f:.4f}) x B{k+1}:")

    if abs(M[n-1, n-1]) < tol:                      # cek pivot terakhir (baris 3 kolom 3), tidak tercakup oleh loop di atas
        raise ValueError("Pivot terakhir nol: tidak ada solusi tunggal.")

    # ---------- Substitusi mundur ----------
    x = np.zeros(n)                                 # siapkan wadaj untuk x1, x2, x3
    for i in range(n - 1, -1, -1):                  # mulai dari baris terakhir naik ke baris pertama (i = 2, 1, 0)
        x[i] = (M[i, n] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
        # M[i, n] = konstanta baris i; np.dot(...) = jumlah koefisien kali variabel yang sudah diketahui; dibagi koefisien x[i]
    return x                                        # kembalikan solusi [x1, x2, x3]

# Cell 4: Data sistem persamaan dan eksekusi
A = [[2, 1, -1],        # persamaan (1): 2x1 + x2 -x3 = 8
     [-3, -1, 2],       # persamaan (2): -3x1 - x2 + 2x3 = -11
     [-2, 1, 2]]        # persamaan (3): -2x1 + x2 + 2x3 = -3
b = [8, -11, -3]        # ruas kanan ketiga persamaan

x = eliminasi_gauss(A, b)                  # jalankan fungsi, hasilnya disimpan di x
print("Solusi sistem:")                    # judul hasil
for i, nilai in enumerate(x, start=1):     # enumerate dengan start=1 agar penomoran x1, x2, x3
    print(f"x{i} = {nilai:.6f}")           # tampilkan 6 angka desimal

# Cell 5: Verifikasi hasil
A_np = np.array(A, dtype=float)                       # A dalam bentuk array desimal
b_np = np.array(b, dtype=float)                       # b dalam bentuk array desimal

residual = A_np @ x - b_np                            # A.x - b; jika solusi benar, hasilnya mendekati nol
print("Residual (A.x - b):", residual)                # tampilkan residual tiap persamaan
print("Residual maksimum  :", np.max(np.abs(residual)))  # selisih terbesar sebagai ukuran galat

x_numpy = np.linalg.solve(A_np, b_np)                 # solusi pembanding dari NumPy
print("Solusi NumPy       :", x_numpy)
print("Sama dengan NumPy? :", np.allclose(x, x_numpy))  # True bila semua elemen sama dalam batas toleransi

# Cell 6: Uji sistem tanpa solusi tunggal
A_gagal = [[1, 1, 1],       # persamaan (1)
           [2, 2, 2],       # persamaan (2) = 2 x persamaan (1), jadi saling bergantung
           [1, -1, 1]]      # persamaan (3)
b_gagal = [6, 12, 2]

try:                                                   # try-except agar program tidak berhenti dengan error mentah
    eliminasi_gauss(A_gagal, b_gagal, tampil=False)    # tampil=False supaya output singkat
except ValueError as pesan:                            # tangkap error dari fungsi
    print("Peringatan:", pesan)                        # tampilkan pesan yang mudah dibaca
