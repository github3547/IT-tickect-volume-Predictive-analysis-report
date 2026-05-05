#  IT Ticket Volume — Predictive Analysis Report

> **ACT Government · Digital & Technology Services**  
> A stakeholder-ready regression model predicting monthly IT ticket volume, with an accompanying professional Excel report.

---

## Project Summary

This project builds a **Multiple Linear Regression model** to forecast IT helpdesk ticket volume using operational drivers such as staff count, remote work percentage, system update frequency, and staff tenure. Results are presented in a **six-sheet Excel workbook** designed for non-technical stakeholders.

This work demonstrates the analytical depth and communication skills required for Data Analyst roles in the ACT Public Sector (entry point ~$110,000/year).

---

## Repository Structure

```
├── it_ticket_regression_model.py   # Main model script (data gen, train, eval, export)
├── build_excel.py                  # Builds the stakeholder Excel report from CSVs
├── IT_Ticket_Predictive_Analysis_Report.xlsx  # Final deliverable (Excel)
├── outputs/
│   ├── raw_data.csv
│   ├── feature_importance.csv
│   ├── quarterly_summary.csv
│   ├── forecast.csv
│   ├── metrics.csv
│   ├── chart_actual_vs_pred.png
│   ├── chart_feature_importance.png
│   ├── chart_quarterly.png
│   ├── chart_forecast.png
│   └── chart_residuals.png
└── README.md
```

---

## Model Details

| Item | Detail |
|---|---|
| **Algorithm** | Ordinary Least Squares (OLS) Multiple Linear Regression |
| **Library** | `scikit-learn` `LinearRegression()` |
| **Dataset** | Synthetic — 48 monthly observations (Jan 2021–Dec 2024) |
| **Train/Test Split** | 80/20 chronological (no data leakage) |
| **Target Variable** | Monthly IT Ticket Volume |

### Features Used

| Feature | Description | Direction |
|---|---|---|
| Remote Work % | % of staff working remotely | ▲ Increases tickets |
| System Updates | Count of monthly update events | ▲ Increases tickets |
| Staff Count | Total headcount | ▲ Increases tickets |
| Prior Month Volume | Previous month's ticket count | ▲ Increases tickets |
| Month Number | Calendar month (seasonality) | ▼ Slight decrease mid-year |
| Avg Tenure (Years) | Organisation-wide average tenure | ▼ Decreases tickets |

### Model Performance

| Metric | Value |
|---|---|
| R² (Train) | 0.9748 |
| R² (Test) | 0.4106 |
| MAE | ~15.7 tickets/month |
| RMSE | ~18.7 tickets/month |

> **Note on R² gap**: The training R² is high due to the synthetic data generation process. In production, the test R² is the critical metric. The model generalises reasonably for a linear baseline and provides directionally correct forecasts.

---

##  Excel Report Structure

The Excel workbook (`IT_Ticket_Predictive_Analysis_Report.xlsx`) contains six sheets:

| Sheet | Contents |
|---|---|
| 📊 Executive Dashboard | KPI cards, embedded charts, key findings & recommendations |
| 📋 Raw Data | Full 48-month dataset with actuals, predictions and residuals |
| 📈 Model Results | Performance metrics, feature importance table, regression equation |
| 📅 Quarterly Summary | Aggregated actuals vs predicted by quarter + totals |
| 🔮 6-Month Forecast | Jan–Jun 2025 forecast with assumptions and confidence ratings |
| 📝 Methodology | Technical notes, limitations, tools used |

---

##  Key Findings

1. **Remote Work %** is the strongest driver — each 1% increase adds ~1.8 tickets/month.  
   → *Scale remote support tooling proportionally with hybrid policy expansion.*

2. **System Updates** generate ~6.3 additional tickets per event.  
   → *Schedule major updates in low-demand periods and pair with pre-release comms.*

3. **Staff Tenure** is negatively correlated (−11.5 per year of tenure).  
   → *Retention and L&D programs will structurally reduce ticket load over time.*

4. **Forecast** projects gradual growth to ~490–510 tickets/month by Jun 2025.  
   → *Consider hiring 1 additional L2 analyst by Q2 2025 to protect SLA compliance.*

---

##  How to Run

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib openpyxl
```

### 2. Run the model (generates CSVs and charts)
```bash
python it_ticket_regression_model.py
```

### 3. Build the Excel report
```bash
python build_excel.py
```

---

## Extending This Project

- **Replace synthetic data** with a real ITSM export (ServiceNow, Jira Service Desk, Freshdesk CSV)
- **Try regularised models** (Ridge, Lasso) if multicollinearity is detected
- **Add SARIMA** for better time-series seasonality capture
- **Automate refresh** by scheduling the scripts via cron or Azure Data Factory
- **Power BI version**: Connect the CSVs to Power BI Desktop for interactive dashboards

---

##  Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
matplotlib
openpyxl
```

---

