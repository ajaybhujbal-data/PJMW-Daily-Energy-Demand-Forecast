<div align="center">

# ⚡ PJMW Daily Energy Demand Forecast

### Hourly electricity demand forecasting for the PJM West region using machine learning

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-189AB4)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![MAPE](https://img.shields.io/badge/MAPE-0.98%25-success)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

**🚀 [Live Demo](https://pjmw-daily-energy-demand-forecast.streamlit.app)** &nbsp;|&nbsp; 📓 [Notebooks](#-project-structure) &nbsp;|&nbsp; 🎤 [Presentation](YOUR-PRESENTATION-LINK)

</div>

---

## 📌 Table of Contents
1. [About the Project](#-about-the-project)
2. [Problem Statement & Objective](#-problem-statement--objective)
3. [Project Journey (4 Weeks)](#-project-journey-4-weeks)
4. [Dataset](#-dataset)
5. [Methodology](#-methodology)
6. [Model Comparison & Results](#-model-comparison--results)
7. [App Features](#-app-features)
8. [Tech Stack](#-tch-stack)
9. [Project Structure](#-project-structure)
10. [How to Run Locally](#-how-to-run-locally)
11. [Limitations & Future Work](#-limitations--future-work)
12. [Team](#-team)
13. [Acknowledgements](#-acknowledgements)

---

## 📖 About the Project

This is our **academic project at ExcelR Academy**, completed as a team of **7 members** over **4 weeks**.

Electricity cannot be stored cheaply at scale, so power companies must predict demand hour by hour. If they predict too low, the grid is at risk. If they predict too high, money and fuel are wasted.

In this project we analysed **15+ years of hourly demand data from PJM West (PJMW)**, built and compared several forecasting models, selected the best one (**Tuned XGBoost**), and deployed it as an interactive **Streamlit web app** that forecasts demand up to **31 days ahead**.

## 🎯 Problem Statement & Objective

**Problem:** Forecast the hourly electricity demand (in MW) of the PJM West region.

**Objectives**
- Explore the data and understand daily, weekly and yearly demand patterns (EDA)
- Build and compare multiple forecasting models
- Choose the most accurate model and save it for reuse
- Deploy it as a user-friendly web application
- Present the results and business value

## 🗓️ Project Journey (4 Weeks)

| Week | Phase | What we did |
|:---:|---|---|
| **1** | 🔍 **EDA** | Cleaned the data, explored trends, seasonality, hourly / weekly / monthly patterns and holidays |
| **2** | 🧠 **Model Building** | Created features and trained 5 model families: ARIMA, RNN, LSTM, Random Forest and XGBoost (plus tuned versions and a baseline) |
| **3** | 🚀 **Deployment** | Saved the best model as a `.pkl` file and built and deployed the Streamlit app |
| **4** | 🎤 **Final Presentation** | Presented the problem, approach, results and live demo |

## 📊 Dataset

| Item | Details |
|---|---|
| **Source** | PJM Interconnection – hourly estimated energy consumption (PJM West region) |
| **File** | `PJMW_MW_Hourly.xlsx` |
| **Columns** | `Datetime` (hourly timestamp), `PJMW_MW` (demand in megawatts) |
| **Size** | 143,206 raw rows, 143,202 after removing duplicate hours (daylight-saving clock changes) |
| **Period** | Dec 2002 – Aug 2018 |

## 🔬 Methodology

```
Raw data  →  Cleaning  →  Feature engineering  →  Train / Test split  →  Model training  →  Evaluation  →  Save .pkl  →  Streamlit app
```

**1. Cleaning** – merged duplicate timestamps (average), sorted by time.

**2. Feature engineering (15 features)**

| Group | Features |
|---|---|
| Calendar | `hour`, `Dayofweek`, `isweekend`, `Is_Holiday`, `month`, `year` |
| Past demand | `lag_1` (1 hour ago), `lag_24` (same hour yesterday), `lag_168` (same hour last week) |
| Rolling averages | `rolling_24`, `rolling_168` |
| Cyclic encoding | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |

**3. Time-based split** – models were trained on the older data and tested on the **last 12 months** (no shuffling, so there is no data leakage).

**4. Forecasting method** – the app uses **recursive hourly forecasting**: each predicted hour is fed back in as the lag / rolling input for the next hour.

## 🏆 Model Comparison & Results

All models were tested on the same last-12-months hold-out set. **Lower is better.**

| Rank | Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|:---:|---|---:|---:|---:|
| 🥇 | **Tuned XGBoost** | **56.50** | 79.25 | **0.97** |
| 🥈 | XGBoost | 56.94 | 79.14 | 0.98 |
| 🥉 | Random Forest | 58.96 | 77.41 | 1.03 |
| 4 | Tuned Random Forest | 59.12 | 77.69 | 1.03 |
| 5 | LSTM | 114.84 | 149.63 | 2.01 |
| 6 | RNN | 230.95 | 355.19 | 3.85 |
| 7 | Baseline (same hour yesterday) | 382.68 | 501.40 | 6.66 |
| 8 | ARIMA | 876.68 | 1137.03 | 14.45 |

**Why XGBoost?** It had the lowest MAE and MAPE, trains fast, and the saved model is small (~8 MB), which makes deployment easy.

> ⚠️ These scores are for **one-step-ahead** prediction (the real previous hour is known). In the app, multi-day forecasts are recursive, so accuracy drops the further ahead you forecast.

## ✨ App Features

- 🎚️ Choose a forecast horizon from **1 to 31 days**
- 🚀 One-click **Generate Forecast** button
- 📈 Hourly forecast chart with peak and lowest demand marked
- 💡 Insight cards: peak demand, lowest demand, estimated energy (GWh)
- 📅 Daily summary table (average, min, max, GWh)
- 🔥 Demand heatmap (day × hour), in the tabbed version
- 🏆 Model comparison and feature-importance charts, in the tabbed version
- ⬇️ Download hourly and daily forecasts as CSV
- 🎨 Custom UI with a renewable-energy background and matching colours

### 📸 Screenshots

> *Add your screenshots here, for example:*
> `![Home](screenshots/home.png)` &nbsp; `![Forecast](screenshots/forecast.png)`

## 🛠️ Tech Stack

| Purpose | Tools |
|---|---|
| Language | Python |
| Data | Pandas, NumPy |
| Modelling | XGBoost, Scikit-learn, Statsmodels (ARIMA), TensorFlow / Keras (RNN, LSTM) |
| Visualisation | Altair, Matplotlib, Seaborn |
| Web app | Streamlit |
| Deployment | Streamlit Community Cloud + GitHub |

## 📁 Project Structure

```
PJMW-Daily-Energy-Demand-Forecast/
│
├── app.py                 # Streamlit app (single Generate button, scroll view)
├── app_tabs.py            # Alternative tabbed version of the app
├── train_model.py         # Trains XGBoost and saves xgb_model.pkl (step-by-step, commented)
├── xgb_model.pkl          # Saved trained model + feature list + metrics
├── PJMW_history.csv       # Cleaned hourly history (starting point for the forecast)
├── images.jpg             # Background image
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Theme settings
└── README.md
```

> Add your **EDA notebook** (Week 1) and **model-building notebook** (Week 2) to a `notebooks/` folder.

## 💻 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/ajaybhujbal-data/PJMW-Daily-Energy-Demand-Forecast.git
cd PJMW-Daily-Energy-Demand-Forecast

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

**Retrain the model from the data** (optional):

```bash
python train_model.py PJMW_MW_Hourly.xlsx
```

This recreates `xgb_model.pkl` and `PJMW_history.csv`.

## 🔭 Limitations & Future Work

- Forecasts use only calendar and past-demand information. **Weather** (temperature especially) is a major driver of demand and is not included yet.
- Recursive forecasting accumulates error over long horizons, so a 31-day forecast is less reliable than a 1-day forecast.
- The data ends in August 2018, so forecasts start from that date.

**Future improvements**
- Add weather data (temperature, humidity) as features
- Add prediction intervals to show forecast uncertainty
- Try hybrid or ensemble models and direct multi-step forecasting
- Automatically refresh the data and retrain on a schedule

## 👥 Team

**Team size:** 7 members &nbsp;|&nbsp; **Institute:** ExcelR Academy

| # | Name | Role / Contribution |
|:---:|---|---|
| 1 | *Your Name* | *e.g. EDA / Model building / Deployment* |
| 2 | *Member name* | *Contribution* |
| 3 | *Member name* | *Contribution* |
| 4 | *Member name* | *Contribution* |
| 5 | *Member name* | *Contribution* |
| 6 | *Member name* | *Contribution* |
| 7 | *Member name* | *Contribution* |

## 🙏 Acknowledgements

- **ExcelR Academy** and our mentors for guidance throughout the 4-week project
- **PJM Interconnection** for the hourly energy consumption data
- Open-source communities behind Streamlit, XGBoost, Pandas and Scikit-learn

---

<div align="center">

⭐ If you like this project, please give it a star! ⭐

</div>
