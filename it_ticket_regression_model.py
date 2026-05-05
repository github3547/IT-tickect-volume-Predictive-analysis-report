#!/usr/bin/env python3
"""
================================================================================
IT Ticket Volume Predictive Analysis — Multiple Linear Regression
================================================================================
Author   : Portfolio Project (ACT Government Context)
Date     : 2025
Dataset  : Synthetic enterprise IT support dataset (48 months, Jan 2021–Dec 2024)
Model    : scikit-learn LinearRegression (OLS)
Outputs  : raw_data.csv, feature_importance.csv, quarterly_summary.csv,
           forecast.csv, metrics.csv, 5x PNG charts
           → Feed into build_excel.py to generate stakeholder Excel report

Usage:
    python it_ticket_regression_model.py

Requirements:
    pip install scikit-learn pandas numpy matplotlib openpyxl
================================================================================
"""

"""
IT Ticket Volume Predictive Analysis
Author: Data Analyst Portfolio Project
Dataset: Synthetic IT Support Ticket Data (based on realistic enterprise patterns)
Model: Multiple Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. Generate realistic synthetic dataset ──────────────────────────────────
months = pd.date_range('2021-01-01', '2024-12-01', freq='MS')
n = len(months)

staff_count       = np.round(200 + np.linspace(0, 80, n) + np.random.normal(0, 5, n)).astype(int)
system_updates    = np.random.poisson(3, n)
avg_tenure_yrs    = np.round(np.clip(4.5 - 0.02*np.arange(n) + np.random.normal(0, 0.3, n), 1.5, 7), 1)
remote_pct        = np.clip(30 + np.linspace(0, 35, n) + np.random.normal(0, 3, n), 20, 80).round(1)
prior_month_vol   = np.zeros(n)

base_tickets = (
    0.8 * staff_count
    + 6.5 * system_updates
    - 4.2 * avg_tenure_yrs
    + 0.9 * remote_pct
    + np.random.normal(0, 18, n)
)
base_tickets = np.clip(base_tickets, 80, None).round().astype(int)

for i in range(1, n):
    prior_month_vol[i] = base_tickets[i-1]

df = pd.DataFrame({
    'Month':            months,
    'Ticket_Volume':    base_tickets,
    'Staff_Count':      staff_count,
    'System_Updates':   system_updates,
    'Avg_Tenure_Years': avg_tenure_yrs,
    'Remote_Pct':       remote_pct,
    'Prior_Month_Vol':  prior_month_vol.astype(int),
})
df['Month_Num']    = df['Month'].dt.month
df['Quarter']      = df['Month'].dt.quarter
df['Year']         = df['Month'].dt.year
df['Month_Label']  = df['Month'].dt.strftime('%b %Y')

# ── 2. Train / test split & model ────────────────────────────────────────────
features = ['Staff_Count', 'System_Updates', 'Avg_Tenure_Years',
            'Remote_Pct', 'Prior_Month_Vol', 'Month_Num']

X = df[features]
y = df['Ticket_Volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)
df['Predicted'] = model.predict(X).round().astype(int)
df['Residual']  = df['Ticket_Volume'] - df['Predicted']

r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test,  y_pred_test)
mae      = mean_absolute_error(y_test, y_pred_test)
rmse     = mean_squared_error(y_test, y_pred_test) ** 0.5

# Feature importance (standardised coefficients)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_std = LinearRegression().fit(X_scaled, y)
importance_df = pd.DataFrame({
    'Feature':    features,
    'Coefficient': model.coef_.round(4),
    'Std_Coeff':   model_std.coef_.round(4),
}).sort_values('Std_Coeff', ascending=False).reset_index(drop=True)

# ── 3. Quarterly summary ──────────────────────────────────────────────────────
quarterly = df.groupby(['Year','Quarter']).agg(
    Actual    = ('Ticket_Volume','sum'),
    Predicted = ('Predicted','sum'),
    Avg_Staff = ('Staff_Count','mean'),
    Updates   = ('System_Updates','sum'),
).round(1).reset_index()
quarterly['Label'] = 'Q' + quarterly['Quarter'].astype(str) + ' ' + quarterly['Year'].astype(str)

# ── 4. 6-month forecast ───────────────────────────────────────────────────────
last = df.iloc[-1]
forecast_rows = []
prev_vol = int(last['Ticket_Volume'])
for i in range(1, 7):
    fmonth = last['Month'] + pd.DateOffset(months=i)
    row = {
        'Month':            fmonth,
        'Month_Label':      fmonth.strftime('%b %Y'),
        'Staff_Count':      int(last['Staff_Count']) + i*3,
        'System_Updates':   3,
        'Avg_Tenure_Years': round(float(last['Avg_Tenure_Years']) - 0.05*i, 1),
        'Remote_Pct':       round(float(last['Remote_Pct']) + 0.5*i, 1),
        'Prior_Month_Vol':  prev_vol,
        'Month_Num':        fmonth.month,
    }
    pred = int(model.predict(pd.DataFrame([row])[features])[0])
    row['Forecast'] = pred
    forecast_rows.append(row)
    prev_vol = pred

forecast_df = pd.DataFrame(forecast_rows)

# ── 5. Export all artefacts ───────────────────────────────────────────────────
df.to_csv('/home/claude/raw_data.csv', index=False)
importance_df.to_csv('/home/claude/feature_importance.csv', index=False)
quarterly.to_csv('/home/claude/quarterly_summary.csv', index=False)
forecast_df.to_csv('/home/claude/forecast.csv', index=False)

metrics = {
    'R2_Train': round(r2_train, 4),
    'R2_Test':  round(r2_test, 4),
    'MAE':      round(mae, 2),
    'RMSE':     round(rmse, 2),
    'Intercept': round(model.intercept_, 4),
    'N_Train':  len(X_train),
    'N_Test':   len(X_test),
}
pd.DataFrame([metrics]).to_csv('/home/claude/metrics.csv', index=False)

# ── 6. Matplotlib charts (saved as PNG for embedding) ────────────────────────
BLUE   = '#1F4E79'
ORANGE = '#C55A11'
GREY   = '#A6A6A6'
GREEN  = '#375623'
LIGHT  = '#D6E4F0'

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)

# Chart 1 – Actual vs Predicted
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['Month'], df['Ticket_Volume'], color=BLUE,   lw=2,   label='Actual',    zorder=3)
ax.plot(df['Month'], df['Predicted'],    color=ORANGE,  lw=1.5, linestyle='--', label='Predicted', zorder=3)
ax.axvline(df['Month'].iloc[len(X_train)], color=GREY, lw=1, linestyle=':', label='Train/Test split')
ax.fill_between(df['Month'], df['Ticket_Volume'], df['Predicted'], alpha=0.08, color=ORANGE)
ax.set_title('Actual vs Predicted IT Ticket Volume (2021–2024)', fontsize=13, fontweight='bold', color=BLUE)
ax.set_ylabel('Ticket Volume', fontsize=10)
ax.legend(fontsize=9)
style_ax(ax)
fig.tight_layout()
fig.savefig('/home/claude/chart_actual_vs_pred.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2 – Feature importance
fig, ax = plt.subplots(figsize=(8, 4))
colors = [BLUE if v >= 0 else ORANGE for v in importance_df['Std_Coeff']]
bars = ax.barh(importance_df['Feature'], importance_df['Std_Coeff'], color=colors, edgecolor='white')
ax.set_title('Feature Importance (Standardised Coefficients)', fontsize=12, fontweight='bold', color=BLUE)
ax.set_xlabel('Standardised Coefficient', fontsize=10)
ax.axvline(0, color='black', lw=0.8)
style_ax(ax)
fig.tight_layout()
fig.savefig('/home/claude/chart_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3 – Quarterly actual vs predicted
fig, ax = plt.subplots(figsize=(14, 4))
x = np.arange(len(quarterly))
w = 0.38
ax.bar(x - w/2, quarterly['Actual'],    width=w, color=BLUE,   label='Actual',    alpha=0.9)
ax.bar(x + w/2, quarterly['Predicted'], width=w, color=ORANGE, label='Predicted', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(quarterly['Label'], rotation=45, ha='right', fontsize=8)
ax.set_title('Quarterly Ticket Volume – Actual vs Predicted', fontsize=12, fontweight='bold', color=BLUE)
ax.set_ylabel('Total Tickets', fontsize=10)
ax.legend(fontsize=9)
style_ax(ax)
fig.tight_layout()
fig.savefig('/home/claude/chart_quarterly.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4 – 6-month forecast
fig, ax = plt.subplots(figsize=(10, 4))
hist_tail = df.tail(6)
ax.plot(hist_tail['Month'], hist_tail['Ticket_Volume'], color=BLUE, lw=2, marker='o', label='Historical (last 6 mo)')
ax.plot(forecast_df['Month'], forecast_df['Forecast'],  color=GREEN, lw=2, marker='s', linestyle='--', label='Forecast (next 6 mo)')
for _, row in forecast_df.iterrows():
    ax.annotate(str(row['Forecast']), (row['Month'], row['Forecast']),
                textcoords='offset points', xytext=(0,8), ha='center', fontsize=8, color=GREEN)
ax.axvline(df['Month'].iloc[-1], color=GREY, lw=1, linestyle=':')
ax.set_title('6-Month Ticket Volume Forecast (Jan–Jun 2025)', fontsize=12, fontweight='bold', color=BLUE)
ax.set_ylabel('Ticket Volume', fontsize=10)
ax.legend(fontsize=9)
style_ax(ax)
fig.tight_layout()
fig.savefig('/home/claude/chart_forecast.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 5 – Residual distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df['Residual'], bins=20, color=BLUE, edgecolor='white', alpha=0.85)
ax.axvline(0, color=ORANGE, lw=2, linestyle='--')
ax.set_title('Residual Distribution (Model Error)', fontsize=12, fontweight='bold', color=BLUE)
ax.set_xlabel('Residual (Actual − Predicted)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
style_ax(ax)
fig.tight_layout()
fig.savefig('/home/claude/chart_residuals.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ All data and charts exported.")
print(f"   R² Test : {r2_test:.4f}")
print(f"   MAE     : {mae:.1f} tickets")
print(f"   RMSE    : {rmse:.1f} tickets")
print(f"   Intercept: {model.intercept_:.2f}")
print("\nCoefficients:")
print(importance_df.to_string(index=False))
