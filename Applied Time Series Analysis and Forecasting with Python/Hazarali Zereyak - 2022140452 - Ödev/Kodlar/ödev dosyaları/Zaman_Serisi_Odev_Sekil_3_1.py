# -*- coding: utf-8 -*-
"""
==============================================================================
PYTHON İLE ZAMAN SERİLERİ ANALİZİ ÖDEV RAPORU
==============================================================================
Ad-Soyad     : Hazarali Zereyak
Okul Numarası: 2022140452
GitHub Linki : https://github.com/hazarrzz
Konu         : Şekil 3.1 - Güney Yarımküre Sıcaklık Dalgalanması Serisi
               (Southtemperature.txt, Ocak 1850 - Aralık 2007, T = 1896 Ay)
==============================================================================
"""

# ============================================================================
# BÖLÜM 1: ZAMAN SERİSİ GRAFİĞİ VE BİLEŞENLERİ (TREND, MHO VE SAÇILIM)
# ============================================================================

# Gerekli kütüphaneleri çalışma ortamına dahil ediyoruz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller

# Southtemperature.txt dosyasındaki aylık verileri matris olarak okuyoruz
temp = np.loadtxt("Southtemperature.txt")

# Yıllara göre satırlarda duran matrisi tek boyutlu vektöre çeviriyoruz
tem = temp.reshape(1, -1)

# 1850 Ocak'tan başlayan 1896 aylık kesikli zaman indeksini oluşturuyoruz
dates = pd.date_range("1850-01", periods=temp.size, freq="M")

# Veriyi ve tarih indeksini birleştirerek Z_t zaman serisini elde ediyoruz
temp_ts = pd.Series(np.ndarray.tolist(tem[0]), index=dates)

# Kitaptaki Şekil 3.1 zaman serisi grafiğini çizdirip kaydediyoruz
fig = plt.figure(figsize=(10, 4.5))
temp_ts.plot(color="#1d4ed8", linewidth=0.8)
plt.title("Şekil 3.1: Güney Yarımküre Sıcaklık Serisi (Z_t, 1850-2007)")
plt.ylabel("Sıcaklık Anomalisi (°C)")
plt.xlabel("Zaman (t)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pyTSA_TemperatureSH_fig3-1.png", dpi=300)
plt.close(fig)

# Ders notlarındaki Merkezsel Hareketli Ortalama (MHO, s=12) hesabı
mho_12 = temp_ts.rolling(window=12, center=True).mean()

# Uzun dönemli trend ve saçılım için 120 aylık MHO ve standart sapma
mho_120 = temp_ts.rolling(window=120, center=True).mean()
roll_std_120 = temp_ts.rolling(window=120, center=True).std()

# Orijinal seri, MHO(12), MHO(120) ve saçılım grafiğini çizdiriyoruz
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
axes[0].plot(temp_ts, color="#94a3b8", lw=0.7, label="Orijinal Seri (Z_t)")
axes[0].plot(mho_12, color="#2563eb", lw=1.2, label="12 Aylık MHO (s=12)")
axes[0].plot(mho_120, color="#dc2626", lw=2.2, label="120 Aylık MHO (Trend)")
axes[0].set_title("Sıcaklık Serisi (Z_t) ve Merkezsel Hareketli Ortalama")
axes[0].set_ylabel("Anomali (°C)")
axes[0].legend(loc="upper left")
axes[0].grid(True, alpha=0.3)

# İkinci panelde sabit varyans şartını (dalga boyunu) inceliyoruz
axes[1].plot(roll_std_120, color="#059669", lw=1.8, label="120 Aylık Std. Sapma")
axes[1].axhline(temp_ts.std(), color="#dc2626", ls="--", label="Genel Std. Sapma")
axes[1].set_title("Zamana Göre Saçılım / Dalga Boyu (Sabit Varyans Kontrolü)")
axes[1].set_ylabel("Standart Sapma")
axes[1].set_xlabel("Yıllar")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_ts_rolling.png", dpi=300)
plt.close(fig)

# Örneklem ortalaması ve varyansı (gamma_0) yazdırıyoruz
print(f"Gözlem Sayısı (T)        : {len(temp_ts)}")
print(f"Örneklem Ortalaması      : {temp_ts.mean():.4f} °C")
print(f"Örneklem Varyansı (γ̂_0)  : {temp_ts.var(ddof=0):.4f}")
print(f"Standart Sapma           : {temp_ts.std():.4f} °C")
print(f"Min / Maks Değerler      : {temp_ts.min():.4f} / {temp_ts.max():.4f}")
print(f"1850-1900 Ortalaması     : {temp_ts['1850':'1900'].mean():.4f} °C")
print(f"1951-2007 Ortalaması     : {temp_ts['1951':'2007'].mean():.4f} °C")


# ============================================================================
# BÖLÜM 2: GECİKME (LAG) İŞLEMCİSİ (B^k Z_t = Z_{t-k})
# ============================================================================

# Ders notlarındaki gecikme işlemcisi B^k Z_t = Z_{t-k} tablosunu kuruyoruz
lag_df = pd.DataFrame({
    "Z_t": temp_ts,
    "Z_{t-1} (B^1)": temp_ts.shift(1),
    "Z_{t-2} (B^2)": temp_ts.shift(2),
    "Z_{t-12} (B^12)": temp_ts.shift(12)
})

# İlk 5 gözlemi yazdırarak gecikme kaymasını ve NaN değerleri görüyoruz
print("\n--- Gecikme İşlemcisi (B^k Z_t = Z_{t-k}) İlk 5 Gözlem ---")
print(lag_df.head(5).round(3))

# Z_t ile gecikmeleri arasındaki otokorelasyon (r_k) katsayılarını buluyoruz
print("\n--- Z_t ile Gecikmeleri Arasındaki Korelasyon (r_k) ---")
for k in [1, 2, 6, 12, 24]:
    r_k = temp_ts.corr(temp_ts.shift(k))
    print(f"Gecikme k={k:2d} (Z_t ile Z_{{t-{k:2d}}}) : r_{k:<2d} = {r_k:.4f}")

# Z_t ile Z_{t-k} arasındaki doğrusal ilişkiyi saçılım grafiğiyle çiziyoruz
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
lags = [1, 2, 6, 12]
for ax, k in zip(axes.flatten(), lags):
    z_lag = temp_ts.shift(k)
    r_val = temp_ts.corr(z_lag)
    ax.scatter(z_lag, temp_ts, alpha=0.35, s=12, color="#2563eb")
    valid = pd.concat([z_lag, temp_ts], axis=1).dropna()
    m, b = np.polyfit(valid.iloc[:, 0], valid.iloc[:, 1], 1)
    x_ln = np.linspace(valid.iloc[:, 0].min(), valid.iloc[:, 0].max(), 100)
    ax.plot(x_ln, m * x_ln + b, color="#dc2626", lw=2)
    ax.set_title(f"Gecikme k = {k} (B^{k} Z_t = Z_{{t-{k}}}) | r_{k} = {r_val:.4f}")
    ax.set_xlabel(f"Z_{{t-{k}}}")
    ax.set_ylabel("Z_t")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_lag_plots.png", dpi=300)
plt.close(fig)


# ============================================================================
# BÖLÜM 3: FARK İŞLEMCİSİ (Delta Z_t = Z_t - Z_{t-1})
# ============================================================================

# 1. Dereceden fark işlemcisi: Delta Z_t = Z_t - Z_{t-1} = (1 - B)Z_t
dt = temp_ts.diff(1)

# İlk gözlem (Z_0 olmadığı için) NaN olduğundan çıkarıyoruz (T = 1895)
dt = dt.dropna()

# 2. Dereceden fark: Delta^2 Z_t = Z_t - 2*Z_{t-1} + Z_{t-2} = (1 - B)^2 Z_t
dt2 = temp_ts.diff(1).diff(1).dropna()

# Mevsimsel fark işlemcisi: Delta_12 Z_t = Z_t - Z_{t-12} = (1 - B^12)Z_t
dt_seas = temp_ts.diff(12).dropna()

# Kitaptaki gibi 1. fark serisini ve 36 aylık hareketli ortalamayı çiziyoruz
fig = plt.figure(figsize=(10, 4.5))
dt.plot(color="#2563eb", lw=0.7, label="1. Fark Serisi (Delta Z_t)")
dt.rolling(36).mean().plot(color="#dc2626", lw=2, label="36 Aylık BHO")
plt.title("Birinci Farkı Alınmış Sıcaklık Serisi (Delta Z_t = Z_t - Z_{t-1})")
plt.ylabel("Fark (°C)")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pyTSA_TemperatureSH_fig3-3.png", dpi=300)
plt.close(fig)

# Orijinal seri, 1. fark ve 2. fark serilerini alt alta karşılaştırıyoruz
fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5), sharex=True)
axes[0].plot(temp_ts, color="#dc2626", lw=0.8)
axes[0].axhline(temp_ts.mean(), color="black", ls="--")
axes[0].set_title("1) Orijinal Seri (Z_t) — Trend Var, Ortalama Sabit Değil")

axes[1].plot(dt, color="#2563eb", lw=0.7)
axes[1].axhline(0, color="black", ls="--")
axes[1].set_title("2) 1. Fark Serisi (Delta Z_t = Z_t - Z_{t-1}) — Trend Yok")

axes[2].plot(dt2, color="#059669", lw=0.6)
axes[2].axhline(0, color="black", ls="--")
axes[2].set_title("3) 2. Fark Serisi (Delta^2 Z_t = Z_t - 2Z_{t-1} + Z_{t-2})")
plt.tight_layout()
plt.savefig("fig_diff_comparison.png", dpi=300)
plt.close(fig)

# Fark alma işlemi öncesi ve sonrası istatistikleri karşılaştırıyoruz
print("\nSeri                                   Gözlem    Ortalama   Varyans   Std.Sapma")
print("-" * 78)
print(f"Orijinal Seri (Z_t)                      {len(temp_ts):4d}   {temp_ts.mean():9.6f}  {temp_ts.var():8.4f}    {temp_ts.std():8.4f}")
print(f"1. Fark (Delta Z_t = Z_t - Z_{{t-1}})      {len(dt):4d}   {dt.mean():9.6f}  {dt.var():8.4f}    {dt.std():8.4f}")
print(f"2. Fark (Delta^2 Z_t)                    {len(dt2):4d}   {dt2.mean():9.6f}  {dt2.var():8.4f}    {dt2.std():8.4f}")
print(f"Mevsimsel Fark (Delta_12 Z_t)            {len(dt_seas):4d}   {dt_seas.mean():9.6f}  {dt_seas.var():8.4f}    {dt_seas.std():8.4f}")


# ============================================================================
# BÖLÜM 4: ACF VE PACF GRAFİKLERİ (BARTLETT, MORAN VE KOK TESTLERİ)
# ============================================================================

# Orijinal serinin (Z_t) ACF ve PACF grafiklerini çizdiriyoruz (m=48 gecikme)
fig = plt.figure(figsize=(10, 6.5))
ax1 = fig.add_subplot(211)
plot_acf(temp_ts, lags=48, ax=ax1, title="Orijinal Seri (Z_t) - ACF (Korelogram)")
ax2 = fig.add_subplot(212)
plot_pacf(temp_ts, lags=48, ax=ax2, title="Orijinal Seri (Z_t) - PACF (KOK)")
plt.tight_layout()
plt.savefig("fig_orig_acf_pacf.png", dpi=300)
plt.close(fig)

# 1. Fark serisinin (Delta Z_t) ACF ve PACF grafiklerini çizdiriyoruz
fig = plt.figure(figsize=(10, 6.5))
ax1 = fig.add_subplot(211)
plot_acf(dt, lags=48, ax=ax1, title="1. Fark Serisi (Delta Z_t) - ACF")
ax2 = fig.add_subplot(212)
plot_pacf(dt, lags=48, ax=ax2, title="1. Fark Serisi (Delta Z_t) - PACF (KOK)")
plt.tight_layout()
plt.savefig("fig_diff_acf_pacf.png", dpi=300)
plt.close(fig)

# Ders notlarındaki Bartlett, Moran ve KOK anlamlılık testlerini uyguluyoruz
T_dt = len(dt)
r_dt = acf(dt, nlags=12, fft=False)
rkk_dt = pacf(dt, nlags=12, method="ywm")

print(f"\n--- 1. Fark Serisi (Delta Z_t, T={T_dt}) ACF & PACF Anlamlılık Testi ---")
print("k     r_k     S_rk(Bartlett)  t_Bartlett  S_rk(Moran)    r_kk   t_PACF   Karar")
print("-" * 78)
for k in [1, 2, 3, 4, 5, 6, 12]:
    # Bartlett standart hatası: S_rk = sqrt((1 + 2 * sum(r_i^2)) / T)
    if k == 1:
        s_bart = np.sqrt(1.0 / T_dt)
    else:
        s_bart = np.sqrt((1.0 + 2.0 * np.sum(r_dt[1:k]**2)) / T_dt)
    t_bart = r_dt[k] / s_bart
    # Moran (SPSS) standart hatası: S_rk = sqrt((T - k) / (T * (T + 2)))
    s_moran = np.sqrt((T_dt - k) / (T_dt * (T_dt + 2)))
    # PACF (KOK) t istatistiği: t = r_kk / (1 / sqrt(T))
    t_pacf = rkk_dt[k] / (1.0 / np.sqrt(T_dt))
    karar = "Anlamlı (H0 Red)" if abs(t_bart) > 1.96 else "Anlamsız (H0 Kabul)"
    print(f"{k:2d}  {r_dt[k]:7.4f}      {s_bart:8.4f}     {t_bart:8.2f}     {s_moran:8.4f}   {rkk_dt[k]:7.4f}  {t_pacf:7.2f}  {karar}")


# ============================================================================
# BÖLÜM 5: DURAĞANLIK ANALİZİ (ADF BİRİM KÖK VE PORTMANTEAU Q TESTLERİ)
# ============================================================================

# 1) ADF Birim Kök Testini Z_t ve Delta Z_t serilerine uyguluyoruz
adf_orig = adfuller(temp_ts)
adf_diff = adfuller(dt)

print("\n=== 1) ADF BİRİM KÖK TESTİ SONUÇLARI ===")
print(f"Orijinal Seri (Z_t)   -> Test İstatistiği: {adf_orig[0]:.4f} | p-değeri: {adf_orig[1]:.4f}")
print(f"1. Fark (Delta Z_t)   -> Test İstatistiği: {adf_diff[0]:.4f} | p-değeri: {adf_diff[1]:.4e}")
print(f"Kritik Değerler (%1, %5, %10): %1={adf_orig[4]['1%']:.3f}, %5={adf_orig[4]['5%']:.3f}, %10={adf_orig[4]['10%']:.3f}")

# 2) Ders notlarındaki Box-Pierce (Q) ve Ljung-Box (Q*) Ak Gürültü testleri
# Box-Pierce: Q = T * sum(r_i^2)  |  Ljung-Box: Q* = T*(T+2)*sum(r_i^2/(T-i))
k_lag = 12
T_o, T_d = len(temp_ts), len(dt)
r_o = acf(temp_ts, nlags=k_lag, fft=False)[1:k_lag+1]
r_d = acf(dt, nlags=k_lag, fft=False)[1:k_lag+1]
lags_arr = np.arange(1, k_lag + 1)

Q_bp_orig = T_o * np.sum(r_o**2)
Q_lb_orig = T_o * (T_o + 2) * np.sum((r_o**2) / (T_o - lags_arr))

Q_bp_diff = T_d * np.sum(r_d**2)
Q_lb_diff = T_d * (T_d + 2) * np.sum((r_d**2) / (T_d - lags_arr))

print(f"\n=== 2) PORTMANTEAU AK GÜRÜLTÜ TESTLERİ (k={k_lag}, Ki-Kare_0.05,12 = 21.03) ===")
print(f"Orijinal Seri (Z_t) -> Box-Pierce Q = {Q_bp_orig:8.2f} | Ljung-Box Q* = {Q_lb_orig:8.2f}")
print(f"1. Fark (Delta Z_t) -> Box-Pierce Q = {Q_bp_diff:8.2f} | Ljung-Box Q* = {Q_lb_diff:8.2f}")

# 3) Ders notlarındaki Şekil A (1. Fark) ve Şekil B (Regresyon) karşılaştırması
t_idx = np.arange(1, len(temp_ts) + 1)
b1, b0 = np.polyfit(t_idx, temp_ts.values, 1)
z_hat = b0 + b1 * t_idx
e_t = pd.Series(temp_ts.values - z_hat, index=temp_ts.index)
adf_reg = adfuller(e_t)
print(f"\n=== 3) REGRESYON ANALİZİ (Z_hat = {b0:.4f} + {b1:.6f}*t) REZİDÜLERİ ===")
print(f"Regresyon Rezidüleri (e_t) -> ADF İstatistiği: {adf_reg[0]:.4f} | p-değeri: {adf_reg[1]:.4e}")
