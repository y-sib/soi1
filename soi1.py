import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_excel("1.xlsx", header=None)
x = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
x = x.to_numpy()
n = len(x)

print("Объём выборки n =", n)

xs = np.sort(x)
print("\nВариационный ряд — первые 10 элементов:")
print(xs[:10])
print("Вариационный ряд — последние 10 элементов:")
print(xs[-10:])

# Основное
xbar = x.mean()
s2   = ((x - xbar) ** 2).mean()
s    = np.sqrt(s2)
xmin, xmax = x.min(), x.max()
R    = xmax - xmin
med  = np.median(x)
d    = np.mean(np.abs(x - med))

def quartile_1(a):
    n = len(a)
    a = np.sort(a)
    if (n / 4) % 1 != 0:
        i = int(np.floor(n / 4)) + 1
    else:
        i = int(n / 4)
    return a[i - 1]

def quartile_3(a):
    n = len(a)
    a = np.sort(a)
    if (n / 4) % 1 != 0:
        i = int(np.floor(n / 4)) + 1
    else:
        i = int(n / 4)
    return a[n - i]

# Квартили
q1 = quartile_1(x)
q3 = quartile_3(x)
q  = q3 - q1
tq = 0.5 * (q1 + q3)
tR = 0.5 * (xmin + xmax)

vals, counts = np.unique(x, return_counts=True)
mode_val = vals[np.argmax(counts)]

# Асимметрия и эксцесс
skew = stats.skew(x, bias=False)
kurt = stats.kurtosis(x, bias=False)

print("\n=== ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ ===")
print(f"Среднее x̄           = {xbar:.3f}")
print(f"Медиана med         = {med:.3f}")
print(f"Мода               = {mode_val:.3f}")
print(f"Минимум / Максимум = {xmin:.3f} / {xmax:.3f}")
print(f"Размах R           = {R:.3f}")
print(f"Дисперсия s²       = {s2:.3f}")
print(f"СКО s              = {s:.3f}")
print(f"Ср. абс. отклонение d = {d:.3f}")
print(f"Q1 / Q3            = {q1:.3f} / {q3:.3f}")
print(f"Разность квартилей q = {q:.3f}")
print(f"Полусумма квартилей tq = {tq:.3f}")
print(f"Полусумма экстремумов tR = {tR:.3f}")
print(f"Асимметрия         = {skew:.3f}")
print(f"Эксцесс (excess)   = {kurt:.3f}")

plt.figure(figsize=(10, 6))
plt.hist(x, bins=15, density=True, alpha=0.6,
         edgecolor="black", label="Гистограмма")
grid = np.linspace(xmin, xmax, 500)
plt.plot(grid, stats.gaussian_kde(x)(grid),
         label="Ядерная оценка плотности")
plt.xlabel("Количество въездных поездок, тыс.")
plt.ylabel("Плотность")
plt.title("Распределение inbound visitors по странам, 2022")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("histogram_density.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 4))
plt.boxplot(x, vert=False)
plt.xlabel("Количество въездных поездок, тыс.")
plt.title("Диаграмма размаха (boxplot) inbound visitors, 2022")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("boxplot.png", dpi=300)
plt.show()

# Выбросы
lower = q1 - 1.5 * q
upper = q3 + 1.5 * q
out_mask = (x < lower) | (x > upper)
print(f"\nВыбросов по правилу 1.5·IQR: {out_mask.sum()}")
if out_mask.sum():
    print("Значения выбросов (по убыванию):")
    print(np.sort(x[out_mask])[::-1][:15])

def chi2_normal(data, k=8):
    xb, sb = data.mean(), data.std(ddof=0)
    edges = stats.norm.ppf(np.linspace(0, 1, k + 1), xb, sb)
    edges[0], edges[-1] = -np.inf, np.inf
    obs, _ = np.histogram(data, bins=edges)
    exp = np.full(k, len(data) / k)
    chi2_stat = np.sum((obs - exp) ** 2 / exp)
    df_ = k - 1 - 2
    p = 1 - stats.chi2.cdf(chi2_stat, df_)
    return chi2_stat, df_, p

print("\n=== Проверка H0: нормальность (исходная шкала) ===")
c2, dff, p = chi2_normal(x)
print(f"χ²_набл = {c2:.3f}, df = {dff}, p = {p:.4f}",
      "→ H0", "НЕ отвергается" if p > 0.05 else "ОТВЕРГАЕТСЯ")

shape, loc, scale = stats.lognorm.fit(x, floc=0)
k = 8
edges = stats.lognorm.ppf(np.linspace(0, 1, k + 1),
                          shape, loc=0, scale=scale)
obs, _ = np.histogram(x, bins=edges)
exp = np.full(k, n / k)
chi2_stat = np.sum((obs - exp) ** 2 / exp)
df_ = k - 2 - 1
p_val = 1 - stats.chi2.cdf(chi2_stat, df_)
crit = stats.chi2.ppf(0.95, df_)

print("\n=== Проверка H0: логнормальное распределение ===")
print("Наблюдаемые частоты:", obs)
print("Ожидаемые частоты:  ", exp)
print(f"χ²_набл = {chi2_stat:.3f}")
print(f"df = {df_}")
print(f"Критическое значение χ²_0.95(df) = {crit:.3f}")
print(f"p-value = {p_val:.4f}")
print("Вывод:",
      "H0 НЕ отвергается (распределение можно считать логнормальным)"
      if chi2_stat < crit
      else "H0 ОТВЕРГАЕТСЯ (логнормальная модель не подходит)")

# Интервалы
alpha = 0.05
s_unb = x.std(ddof=1) 

t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
ci_mean_left  = xbar - t_crit * s_unb / np.sqrt(n)
ci_mean_right = xbar + t_crit * s_unb / np.sqrt(n)

chi2_low  = stats.chi2.ppf(alpha / 2, n - 1)
chi2_high = stats.chi2.ppf(1 - alpha / 2, n - 1)
ci_sigma_left  = s_unb * np.sqrt((n - 1) / chi2_high)
ci_sigma_right = s_unb * np.sqrt((n - 1) / chi2_low)

print("\n=== 95%-е ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ===")
print(f"Для среднего μ:     ({ci_mean_left:.3f}; {ci_mean_right:.3f})")
print(f"Для СКО σ:          ({ci_sigma_left:.3f}; {ci_sigma_right:.3f})")

# Сравнение
df_sort = pd.DataFrame({"Значение": x}).sort_values("Значение", ascending=False)
top10 = df_sort.head(10)["Значение"]
bot10 = df_sort.tail(10)["Значение"]
print("\n=== Сравнение крайних групп (для п.4–5) ===")
print("Среднее top-10:   ", round(top10.mean(), 2))
print("Среднее bottom-10:", round(bot10.mean(), 2))
print("Отношение top/bottom:", round(top10.mean() / max(bot10.mean(), 1e-9), 1))

plt.figure(figsize=(10, 6))
plt.barh(["Spain", "United States of America", "Türkiye", "Italy", "Mexico", "United Kingdom of Great Britain and Northern Ireland", "Germany", "Greece", "Austria", "Portugal"][::-1],
         top10.values[::-1])

plt.xlabel("Количество въездных поездок, тыс.")
plt.title("10 стран с наибольшим inbound-потоком, 2022")
plt.tight_layout()
plt.savefig("top10.png", dpi=300)
plt.show()