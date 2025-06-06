import numpy as np
from scipy import stats

mu_0 = 10.23       # Hipotesis rata-rata populasi (%)
x_bar = 11.39      # Rata-rata sampel (%)
s = 2.09           # Standar deviasi sampel (%)
n = 36             # Ukuran sampel

# Hitung nilai t
t_statistic = (x_bar - mu_0) / (s / np.sqrt(n))

# Derajat kebebasan
df = n - 1

# Hitung p-value untuk uji dua sisi
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

# Tampilkan hasil
print(f"Nilai t-statistic: {t_statistic:.4f}")
print(f"Derajat kebebasan: {df}")
print(f"P-value: {p_value:.4f}")