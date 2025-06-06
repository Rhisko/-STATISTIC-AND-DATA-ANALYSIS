
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import datetime

# 1. Membaca Dataset
df = pd.read_csv('data_penjualan.csv')

# 2. Statistik Deskriptif
desc = df.describe().T
print(f"Statistik Deskriptif: \n {desc}\n")


# 3. Model Regresi Linier Berganda
X = df[['Jumlah_Pelanggan', 'Pengeluaran_Iklan', 'Tingkat_Diskon']]
y = df['Pendapatan_Harian']
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
reg_summary = model.summary()

print(f"Ringkasan Model Regresi: \n {reg_summary}\n")

# 4. Uji t, Uji F, Interpretasi Koefisien
params = model.params
pvalues = model.pvalues

print("Koefisien Model:")
for feature, coef, pval in zip(X.columns, params[1:], pvalues[1:]):
    print(f"{feature}: Koefisien = {coef:.4f}, p-value = {pval:.4f}")
# Interpretasi Uji t
for feature, pval in zip(X.columns, pvalues[1:]):
    if pval < 0.05:
        print(f"{feature} berpengaruh signifikan terhadap Pendapatan Harian (p-value = {pval:.4f})")
    else:
        print(f"{feature} tidak berpengaruh signifikan terhadap Pendapatan Harian (p-value = {pval:.4f})")
# Interpretasi Uji F
if model.f_pvalue < 0.05:
    print(f"Model secara keseluruhan signifikan (p-value = {model.f_pvalue:.4f})")
    

# 5. Uji Normalitas Residual
residuals = model.resid
shapiro_stat, shapiro_p = shapiro(residuals)

print(f"Uji Normalitas Residual (Shapiro-Wilk): Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print("Residual tidak terdistribusi normal.")
else:
    print("Residual terdistribusi normal.")

# 6. Uji Multikolinearitas (VIF)
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(f"Uji Multikolinearitas (VIF): \n{vif_data}\n")
print("Jika VIF > 10, ada indikasi multikolinearitas.")
print(vif_data[vif_data['VIF'] > 10])
# Interpretasi VIF
for index, row in vif_data.iterrows():
    if row['VIF'] > 10:
        print(f"{row['Feature']} menunjukkan multikolinearitas (VIF = {row['VIF']:.2f})")
    else:
        print(f"{row['Feature']} tidak menunjukkan multikolinearitas (VIF = {row['VIF']:.2f})")


# 7. Uji Heteroskedastisitas (Breusch-Pagan)
bp_test = het_breuschpagan(residuals, X_const)
bp_labels = ['LM Statistic', 'p-value', 'F-value', 'F p-value']
bp_result = dict(zip(bp_labels, bp_test))

print(f"Uji Heteroskedastisitas (Breusch-Pagan): \n{bp_result}\n")
if bp_result['p-value'] < 0.05:
    print("Ada indikasi heteroskedastisitas (p-value < 0.05)")
else:
    print("Tidak ada indikasi heteroskedastisitas (p-value >= 0.05)")


# 8. Uji Autokorelasi (Durbin-Watson)
dw_stat = sm.stats.stattools.durbin_watson(residuals)
print(f"Statistik Durbin-Watson: {dw_stat:.4f}")

