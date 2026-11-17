import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 2

def bisection(f, a, b, tol=1e-6, maks_iter=100):
    if a >= b:
        raise ValueError("Batas bawah a harus lebih kecil dari batas atas b.")
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) dan f(b) harus berbeda tanda, akar belum terkurung.")

    riwayat = []
    for i in range(1, maks_iter + 1):
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