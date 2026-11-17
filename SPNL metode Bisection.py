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
        c = (a + b) / 2
        fc = f(c)
        galat = (b - a) / 2
        riwayat.append({"Iterasi": i, "a": a, "b": b, "c": c,
                        "f(a)": fa, "f(b)": fb, "f(c)": fc, "Galat": galat})
        if fc == 0 or galat < tol:
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, pd.DataFrame(riwayat)

pd.options.display.float_format = "{:.6f}".format
akar, tabel = bisection(f, a=1, b=2, tol=1e-6)
display(tabel)
print(f"Akar hampiran = {akar:.6f} setelah {len(tabel)} iterasi")

akar_np = [r.real for r in np.roots([1, 0, -1, -2]) if abs(r.imag) < 1e-9][0]
print("Akar pembanding (np.roots):", akar_np)
print("Selisih dengan Bisection  :", abs(akar - akar_np))
print("f(akar)                   :", f(akar))

n_teori = int(np.ceil(np.log2((2 - 1) / 1e-6)))
print("Iterasi teoritis:", n_teori, "| iterasi program:", len(tabel))

xs = np.linspace(0.5, 2.5, 200)
plt.figure(figsize=(7, 4))
plt.plot(xs, f(xs), label=r"$f(x) = x^3 - x - 2$")
plt.axhline(0, color="black", linewidth=0.8)
plt.scatter([akar], [f(akar)], color="red", zorder=3, label=f"akar ≈ {akar:.6f}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Grafik fungsi dan akar hasil Bisection")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

try:
    bisection(f, a=2, b=3)
except ValueError as pesan:
    print("Peringatan:", pesan)
