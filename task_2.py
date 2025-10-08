# task_2.py
import numpy as np
from math import sqrt
from scipy.integrate import quad
from time import perf_counter


# --- 1) функція і межі інтегрування ----
def f(x: np.ndarray) -> np.ndarray:
    """Підінтегральна функція: f(x) = x^2."""
    return x**2

a, b = 0.0, 2.0   # інтегруємо від 0 до 2


# --- 2) інтеграл методом Монте-Карло ---
def monte_carlo_integral(func, a, b, n_samples=100_000, seed=42):
    """
    ∫_a^b f(x) dx ≈ (b-a) * mean( f(U) ), де U ~ U(a,b)
    Повертає (estimate, stderr), де stderr — стандартна помилка ~ O(1/sqrt(N)).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n_samples)
    fx = func(x)

    mean_fx = fx.mean()
    std_fx = fx.std(ddof=1)
    estimate = (b - a) * mean_fx
    stderr   = (b - a) * std_fx / sqrt(n_samples)
    return estimate, stderr


# --- 3) базове обчислення ----------------------------------------------------
N = 200_000  # к-сть випадкових точок (можна змінювати)

# Монте-Карло + час
t0 = perf_counter()
mc_value, mc_err = monte_carlo_integral(f, a, b, n_samples=N, seed=1)
t1 = perf_counter()
mc_time_ms = (t1 - t0) * 1000

# Аналітика + час (для повноти, хоч це миттєво)
t2 = perf_counter()
analytic = (b**3 - a**3) / 3.0   # ∫ x^2 dx = (b^3 - a^3)/3
t3 = perf_counter()
analytic_time_ms = (t3 - t2) * 1000

# SciPy quad + час
t4 = perf_counter()
quad_val, quad_err = quad(lambda t: t**2, a, b)
t5 = perf_counter()
quad_time_ms = (t5 - t4) * 1000

print(f"Метод Монте-Карло (N={N}):   {mc_value:.8f} ± {mc_err:.8f} (time: {mc_time_ms:.3f} ms)")
print(f"Аналітично:                  {analytic:.8f}  (очікуване 8/3 ≈ 2.66666667) (time: {analytic_time_ms:.3f} ms)")
print(f"SciPy quad:                  {quad_val:.8f} ± {quad_err:.2e} (time: {quad_time_ms:.3f} ms)")


# --- 4) Бенчмарк O(N) для Монте-Карло ---------------------------------------
def bench_mc(func, a, b):
    """
    Демонструє практичну лінійність часу від N (O(N)) і збіжність похибки ~1/sqrt(N).
    Виводить час, час/семпл, оцінку, стандартну помилку, 95% ДІ та |Δ| до аналітики.
    """
    Ns = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

    print("\nBenchmark Monte Carlo (O(N))")
    print("N\t\t time_ms\t per_sample_ns\t estimate\t stderr\t\t 95% CI covers exact\t |Δ|")
    print("-"*110)

    # розминка
    monte_carlo_integral(func, a, b, n_samples=2_000, seed=1)
    analytic = (b**3 - a**3) / 3.0

    for N in Ns:
        t0 = perf_counter()
        est, se = monte_carlo_integral(func, a, b, n_samples=N, seed=1)
        t1 = perf_counter()
        dt = t1 - t0
        per_sample_ns = dt * 1e9 / N
        lo, hi = est - 1.96 * se, est + 1.96 * se
        abs_err = abs(est - analytic)
        covers = (lo <= analytic <= hi)
        print(f"{N:<8}\t {dt*1000:8.3f}\t {per_sample_ns:13.1f}\t {est:.8f}\t {se:.6f}\t {str(covers):>7}\t\t {abs_err:.6g}")

# запустити бенчмарк
bench_mc(f, a, b)
