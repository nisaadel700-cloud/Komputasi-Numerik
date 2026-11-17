# Cell 1: Import library
import numpy as np               # operasi numerik: log2, ceil, linspace, np.roots
import pandas as pd              # menyimpan dan menampilkan tabel iterasi
import matplotlib.pyplot as plt  # membuat grafik fungsi dan posisi akar

# Cell 2: Mendefinisikan fungsi f(x)
def f(x):                  # f menerima satu nilai x (boleh juga array NumPy)
    return x**3 - x - 2    # f(x) = x^3 - x - 2, akarnya dicari lewat f(x) = 0

# Cell 3: Fungsi metode Bisection
def bisection(f, a, b, tol=1e-6, maks_iter=100):   # tol = batas galat, maks_iter = pengaman agar tidak berputar tanpa henti
    if a >= b:                                      # batas bawah harus lebih kecil dari batas atas
        raise ValueError("Batas bawah a harus lebih kecil dari batas atas b.")
    fa, fb = f(a), f(b)                             # nilai fungsi di ujung selang, dihitung sekali di awal
    if fa * fb > 0:                                 # hasil kali positif berarti tanda sama, akar tidak terjamin ada di selang
        raise ValueError("f(a) dan f(b) harus berbeda tanda, akar belum terkurung.")

    riwayat = []                                    # daftar kosong untuk menyimpan data tiap iterasi
    for i in range(1, maks_iter + 1):               # iterasi dari 1 sampai maks_iter
        c = (a + b) / 2                             # titik tengah selang saat ini
        fc = f(c)                                   # nilai fungsi di titik tengah
        galat = (b - a) / 2                         # setengah lebar selang dipakai sebagai batas galat
        riwayat.append({"Iterasi": i, "a": a, "b": b, "c": c, "f(a)": fa, "f(b)": fb, "f(c)": fc, "Galat": galat})
        # satu baris data iterasi disamping agar nanti bisa ditampilkan sebagai tabel
        if fc == 0 or galat < tol:                  # berhenti jika c tepat akar atau selang sudah cukup sempit
            break
        if fa * fc < 0:                             # tanda berbeda di a dan c berarti akar ada di [a, c]
            b, fb = c, fc                           # batas atas dipindah ke c
        else:                                       # selain itu akar ada di [c, b]
            a, fa = c, fc                           # batas bawah dipindah ke c
    return c, pd.DataFrame(riwayat)                 # kembalikan akar hampiran dan tabel seluruh iterasi

# Cell 4: Menjalankan Bisection
pd.options.display.float_format = "{:.6f}".format      # semua angka tabel ditampilkan 6 desimal
akar, tabel = bisection(f, a=1, b=2, tol=1e-6)        # selang awal [1, 2] dipilih karena f(1) = -2 dan f(2) = 4 berbeda tanda
display(tabel)                                        # tampilkan tabel iterasi di Colab
print(f"Akar hampiran = {akar:.6f} setelah {len(tabel)} iterasi")  # ringkasan hasil

# Cell 5: Verifikasi hasil
akar_np = [r.real for r in np.roots([1, 0, -1, -2]) if abs(r.imag) < 1e-9][0]
# np.roots menerima koefisien polinom x^3 + 0x^2 - x - 2; hanya satu akar yang real, sisanya kompleks sehingga disaring lewat imag
print("Akar pembanding (np.roots):", akar_np)         # akar menurut NumPy
print("Selisih dengan Bisection  :", abs(akar - akar_np))  # seharusnya lebih kecil dari toleransi
print("f(akar)                   :", f(akar))         # nilai fungsi di akar hampiran, mendekati nol

n_teori = int(np.ceil(np.log2((2 - 1) / 1e-6)))       # n >= log2((b- a) / tol), dibulatkan ke atas
print("Iterasi teoritis:", n_teori, "| iterasi program:", len(tabel))

# Cell 6: Grafik fungsi dan akar
xs = np.linspace(0.5, 2.5, 200)                       # 200 titik x merata dari 0,5 sampai 2,5
plt.figure(figsize=(7, 4))                            # ukuran gambar (lebar, tinggi) dalam inci
plt.plot(xs, f(xs), label=r"$f(x) = x^3 - x - 2$")    # kurva fungsi, f(xs) dihitung sekaligus untuk semua titik
plt.axhline(0, color="black", linewidth=0.8)          # garis sumbu x sebagai acuan f(x) = 0
plt.scatter([akar], [f(akar)], color="red", zorder=3, label=f"akar ≈ {akar:.6f}")  # titik akar, zorder=3 agar di atas garis
plt.xlabel("x")                                       # label sumbu x
plt.ylabel("f(x)")                                    # label sumbu y
plt.title("Grafik fungsi dan akar hasil Bisection")   # judul grafik
plt.legend()                                          # tampilkan keterangan garis dan titik
plt.grid(True, alpha=0.3)                             # grid tipis untuk memudahkan membaca nilai
plt.show()                                            # render grafik

# Cell 7: Uji selang yang tidak valid
try:                                                  # try-except agar error ditangkap dengan pesan yang ramah
    bisection(f, a=2, b=3)                            # f(2) dan f(3) sama-sama positif, syarat Bisection tidak terpenuhi
except ValueError as pesan:                           # tangkap error dari fungsi bisection
    print("Peringatan:", pesan)                       # tampilkan pesan peringatan
