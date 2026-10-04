import base64
import pickle
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

st.set_page_config(page_title="PJMW Daily Energy Demand Forecast", page_icon="⚡", layout="wide")

MAX_DAYS = 31
APP_TITLE = "⚡ PJMW Daily Energy Demand Forecast"   # <- change the project name here


# ---------------------------------------------------------------- theme (colors taken from images.jpg)
SKY, NAVY, GRASS, SUN = "#2f80c9", "#14365d", "#5a9a2f", "#f5b21e"


def set_background(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,25,60,0.18), rgba(0,25,60,0.18)),
                              url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* NO big white box: image is visible behind the whole page */
        .block-container {{ background: transparent; padding-top: 3rem; }}

        [data-testid="stSidebar"] > div:first-child {{
            background: linear-gradient(180deg, rgba(20,54,93,0.88), rgba(47,128,201,0.80));
            backdrop-filter: blur(4px);
        }}
        [data-testid="stSidebar"] * {{ color: #ffffff !important; }}
        [data-testid="stSidebar"] [data-testid="stAlert"] {{ background: rgba(255,255,255,0.18); }}

        /* headings / captions sit directly on the image -> white with shadow */
        h1, h2, h3, h4, .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * {{
            color: #ffffff !important;
            text-shadow: 0 2px 8px rgba(0,20,50,0.75);
        }}
        .stApp [data-testid="stCaptionContainer"] {{ opacity: 1 !important; font-weight: 600; }}
        .block-container [data-testid="stAlert"] {{
            background: rgba(255,255,255,0.93); border-radius: 12px;
        }}
        h1 {{ border-bottom: 4px solid {SUN}; padding-bottom: .3rem; }}
        hr {{ border-color: rgba(255,255,255,0.5) !important; }}

        /* small cards only behind the data itself */
        [data-testid="stExpander"] {{
            background: rgba(255,255,255,0.92); border-radius: 12px; border: none;
        }}
        [data-testid="stExpander"] * {{ text-shadow: none !important; color: {NAVY} !important; }}
        [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stArrowVegaLiteChart"], .stVegaLiteChart {{
            background: rgba(255,255,255,0.92); border-radius: 12px; padding: 6px;
            box-shadow: 0 4px 16px rgba(0,20,50,0.3);
        }}

        /* metrics as cards */
        [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.85);
            border-left: 5px solid {SKY};
            border-radius: 12px;
            padding: 12px 16px;
            box-shadow: 0 2px 10px rgba(20,54,93,0.12);
        }}
        [data-testid="stMetricValue"] {{ color: {NAVY}; }}


        /* box headings (metric labels) -> dark navy, readable on white cards */
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
        [data-testid="stMetric"] label, [data-testid="stMetric"] label * {{
            color: {NAVY} !important; font-weight: 700 !important; text-shadow: none !important; opacity: 1 !important;
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{ color: {NAVY} !important; text-shadow: none !important; }}
        /* green "model loaded" box text -> dark green */
        .block-container [data-testid="stAlert"] * {{ color: #1b5e20 !important; text-shadow: none !important; font-weight: 600; }}

        /* buttons */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, {SKY}, {GRASS});
            border: none; color: #fff; font-weight: 700; border-radius: 12px;
        }}
        .stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); }}
        .stDownloadButton > button {{
            background: {NAVY}; color: #fff; border: none; border-radius: 12px;
        }}
        .stDownloadButton > button:hover {{ background: {SKY}; color: #fff; }}

        .insight {{
            border-radius: 14px; padding: 16px 20px; color: {NAVY};
            box-shadow: 0 2px 10px rgba(20,54,93,0.15);
        }}
        .insight .t {{ font-weight: 700; font-size: 1.05rem; }}
        .insight .v {{ font-size: 1.7rem; font-weight: 800; margin: 4px 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def insight_card(col, title, value, sub, bg, border):
    col.markdown(
        f'<div class="insight" style="background:{bg};border-left:6px solid {border};">'
        f'<div class="t">{title}</div><div class="v">{value}</div><div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


set_background("images.jpg")


def line_chart(df, x, y, color, x_type="T"):
    """Altair line chart whose axes fit the data tightly."""
    chart = (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=2.5)
        .encode(
            x=alt.X(f"{x}:{x_type}", title=None),
            y=alt.Y(f"{y}:Q", title="MW", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip(f"{x}:{x_type}"), alt.Tooltip(f"{y}:Q", format=",.0f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, width="stretch")



# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_history():
    df = pd.read_csv("PJMW_history.csv", parse_dates=["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)


# ---------------------------------------------------------------- forecasting
def recursive_forecast(bundle, history, hours):
    """Predict hour by hour; every prediction is fed back as lag/rolling input."""
    model, features = bundle["model"], bundle["features"]

    s = history.set_index("Datetime")["PJMW_MW"].astype(float)
    tail = s.iloc[-400:]                       # enough for lag_168 / rolling_168
    values = dict(zip(tail.index, tail.values))
    recent = list(tail.values[-168:])          # rolling window (row based, like training)
    last_ts = s.index[-1]

    future = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=hours, freq="h")
    hol = set(USFederalHolidayCalendar().holidays(future.min().normalize(), future.max().normalize()))

    rows, preds = [], []
    for ts in future:
        lag_1 = recent[-1]
        lag_24 = values.get(ts - pd.Timedelta(hours=24), recent[-24])
        lag_168 = values.get(ts - pd.Timedelta(hours=168), recent[-168])
        x = {
            "hour": ts.hour,
            "Dayofweek": ts.dayofweek,
            "isweekend": int(ts.dayofweek >= 5),
            "Is_Holiday": int(ts.normalize() in hol),
            "month": ts.month,
            "year": ts.year,
            "lag_1": lag_1,
            "lag_24": lag_24,
            "lag_168": lag_168,
            "rolling_24": float(np.mean(recent[-24:])),
            "rolling_168": float(np.mean(recent[-168:])),
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
        }
        y = float(model.predict(pd.DataFrame([x])[features])[0])
        values[ts] = y
        recent.append(y)
        recent = recent[-168:]
        preds.append(y)

    return pd.DataFrame({"Datetime": future, "Predicted_MW": preds})


# ---------------------------------------------------------------- UI
try:
    bundle = load_model()
    history = load_history()
except FileNotFoundError:
    st.error("xgb_model.pkl / PJMW_history.csv not found. Run `python train_model.py PJMW_MW_Hourly.xlsx` first.")
    st.stop()

last_ts = history["Datetime"].iloc[-1]
last_val = history["PJMW_MW"].iloc[-1]

with st.sidebar:
    st.header("⚙️ Forecast Settings")
    days = st.slider("Forecast horizon (days)", 1, MAX_DAYS, 10)
    hours = days * 24
    st.info(f"🗓️ The model will forecast the next {days} day(s) ({hours} hours).")
    st.divider()
    st.markdown("🤖 **Model:** XGBoost Regressor")
    st.markdown(f"🧩 **Features:** {len(bundle['features'])}")
    st.markdown(f"📚 **History available:** {len(history):,} rows")
    st.markdown(f"⏱️ **Forecast horizon:** {hours} hours")

st.title(APP_TITLE)
st.caption("PJM West electricity demand forecasting using the trained (tuned) XGBoost model.")
st.success(f"✅ Trained XGBoost model loaded successfully. You can forecast up to {MAX_DAYS} days ahead.")

c1, c2, c3 = st.columns(3)
c1.metric("📊 Historical Rows", f"{len(history):,}")
c2.metric("🕒 Last Timestamp", last_ts.strftime("%Y-%m-%d %H:%M"))
c3.metric("🔌 Last Demand", f"{last_val:,.0f} MW")

st.divider()

if st.button("🚀 Generate Forecast", type="primary", width="stretch"):
    with st.spinner(f"⏳ Generating {days} day forecast..."):
        st.session_state["fc"] = recursive_forecast(bundle, history, hours)
        st.session_state["fc_days"] = days

if "fc" in st.session_state:
    fc = st.session_state["fc"]
    d = st.session_state["fc_days"]
    p = fc["Predicted_MW"]

    st.subheader(f"📈 Next {d} Day(s) Forecast")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("▶️ First Forecast", f"{p.iloc[0]:,.0f} MW")
    k2.metric("🔺 Maximum Forecast", f"{p.max():,.0f} MW")
    k3.metric("🔻 Minimum Forecast", f"{p.min():,.0f} MW")
    k4.metric("⚖️ Average Demand", f"{p.mean():,.0f} MW")

    st.subheader("💡 Forecast Insights")
    i1, i2, i3 = st.columns(3)
    pk, lo = fc.loc[p.idxmax()], fc.loc[p.idxmin()]
    insight_card(i1, "🔥 Peak Demand", f"{pk.Predicted_MW:,.0f} MW", f"{pk.Datetime:%Y-%m-%d %H:%M}", "#fff1cc", SUN)
    insight_card(i2, "❄️ Lowest Demand", f"{lo.Predicted_MW:,.0f} MW", f"{lo.Datetime:%Y-%m-%d %H:%M}", "#dcecfa", SKY)
    insight_card(i3, "🔋 Estimated Energy", f"{p.sum() / 1000:,.2f} GWh", "Over the selected forecast period", "#e3f1d6", GRASS)

    st.subheader("🕐 Hourly Demand Forecast")
    line_chart(fc, "Datetime", "Predicted_MW", SKY)

    daily = (
        fc.assign(Date=fc["Datetime"].dt.date)
        .groupby("Date")["Predicted_MW"]
        .agg(Average_MW="mean", Minimum_MW="min", Maximum_MW="max", Total_GWh=lambda x: x.sum() / 1000)
        .round(2)
        .reset_index()
    )

    st.subheader("📅 Daily Forecast Summary")
    st.dataframe(daily, width="stretch", hide_index=True)

    st.subheader("📉 Daily Average Demand")
    line_chart(daily.assign(Date=pd.to_datetime(daily["Date"])), "Date", "Average_MW", GRASS)

    st.subheader("📋 Hourly Forecast Table")
    hourly_out = fc.copy()
    hourly_out["Predicted_MW"] = hourly_out["Predicted_MW"].round(2)
    hourly_out["Predicted_GW"] = (hourly_out["Predicted_MW"] / 1000).round(3)
    st.dataframe(hourly_out, width="stretch", hide_index=True)

    st.download_button("⬇️ Download Hourly Forecast CSV", hourly_out.to_csv(index=False).encode(),
                       f"pjmw_{d}_day_forecast.csv", "text/csv", width="stretch")
    st.download_button("⬇️ Download Daily Summary CSV", daily.to_csv(index=False).encode(),
                       f"pjmw_{d}_day_daily_summary.csv", "text/csv", width="stretch")

    if d > 14:
        st.caption("⚠️ Note: this is a recursive forecast, so uncertainty grows the further ahead you go.")

with st.expander("ℹ️ Model Information"):
    m = bundle["metrics"]
    st.markdown(
        f"""
**Model:** Tuned XGBoost Regressor  
**Prediction target:** PJMW_MW (MW)  
**Input upload required:** No  
**Historical data:** Bundled PJMW_MW_Hourly data  
**Forecast method:** Recursive hourly forecasting  

**Hold-out accuracy (last 12 months, 1-step-ahead):** MAE {m['MAE']:.1f} MW · RMSE {m['RMSE']:.1f} MW · MAPE {m['MAPE_%']:.2f}%
"""
    )
    st.markdown("#### 🧠 Features used by the model")
    st.table(pd.DataFrame({
        "Feature": bundle["features"],
        "Description": [
            "Hour of day", "Day of week", "Weekend indicator", "US federal holiday", "Month", "Year",
            "Demand 1 hour earlier", "Demand 24 hours earlier", "Demand 168 hours earlier",
            "24-hour rolling average", "168-hour rolling average",
            "Hour (sine)", "Hour (cosine)", "Day of week (sine)", "Day of week (cosine)",
        ],
    }))

st.divider()
st.caption("⚡ PJMW Daily Energy Demand Forecasting · XGBoost · Hourly Recursive Forecast")