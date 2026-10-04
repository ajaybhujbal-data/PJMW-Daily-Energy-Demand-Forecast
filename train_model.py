"""
Train the Tuned XGBoost model (same params as TrainedModels.ipynb) and save it as a .pkl

Usage:
    python train_model.py PJMW_MW_Hourly.xlsx
Outputs:
    xgb_model.pkl          -> model + feature list + metrics (used by app.py)
    PJMW_history.csv       -> cleaned hourly history (used by app.py for lags)
"""
import sys, pickle
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

FEATURES = [
    "hour", "Dayofweek", "isweekend", "Is_Holiday", "month", "year",
    "lag_1", "lag_24", "lag_168", "rolling_24", "rolling_168",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
# Tuned XGBoost params from the notebook (RandomizedSearchCV result)
XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.1, max_depth=7,
                  subsample=0.8, colsample_bytree=1.0, random_state=42, n_jobs=-1)


def load_clean(path):
    df = pd.read_excel(path) if path.endswith(("xlsx", "xls")) else pd.read_csv(path)
    df = df[["Datetime", "PJMW_MW"]].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    # duplicate timestamps (DST fall-back hour) -> average, same as notebook
    df = df.groupby("Datetime", as_index=False)["PJMW_MW"].mean().sort_values("Datetime")
    return df.reset_index(drop=True)


def build_features(df):
    df = df.copy()
    dt = df["Datetime"]
    df["hour"] = dt.dt.hour
    df["Dayofweek"] = dt.dt.dayofweek
    df["isweekend"] = (df["Dayofweek"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    hol = USFederalHolidayCalendar().holidays(dt.min().normalize(), dt.max().normalize() + pd.Timedelta(days=40))
    df["Is_Holiday"] = dt.dt.normalize().isin(hol).astype(int)
    df["lag_1"] = df["PJMW_MW"].shift(1)
    for h in (24, 168):  # timestamp-based lags (robust to DST gaps)
        s = df[["Datetime", "PJMW_MW"]].copy()
        s["Datetime"] += pd.Timedelta(hours=h)
        df = df.merge(s.rename(columns={"PJMW_MW": f"lag_{h}"}), on="Datetime", how="left")
    df["rolling_24"] = df["PJMW_MW"].shift(1).rolling(24).mean()
    df["rolling_168"] = df["PJMW_MW"].shift(1).rolling(168).mean()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["Dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["Dayofweek"] / 7)
    return df.dropna().reset_index(drop=True)


def main(path):
    raw = load_clean(path)
    raw.to_csv("PJMW_history.csv", index=False)
    data = build_features(raw)

    test_start = data["Datetime"].max() - pd.DateOffset(years=1)
    train, test = data[data["Datetime"] < test_start], data[data["Datetime"] >= test_start]

    # 1) honest evaluation: train on past, test on last year
    m = XGBRegressor(**XGB_PARAMS).fit(train[FEATURES], train["PJMW_MW"])
    p = m.predict(test[FEATURES])
    metrics = {
        "MAE": float(mean_absolute_error(test["PJMW_MW"], p)),
        "RMSE": float(np.sqrt(mean_squared_error(test["PJMW_MW"], p))),
        "MAPE_%": float(mean_absolute_percentage_error(test["PJMW_MW"], p) * 100),
    }
    print("Hold-out (last 12 months) one-step-ahead metrics:", metrics)

    # 2) final model: refit on ALL data so it knows the most recent year
    final = XGBRegressor(**XGB_PARAMS).fit(data[FEATURES], data["PJMW_MW"])

    bundle = {
        "model": final,
        "features": FEATURES,
        "metrics": metrics,
        "last_timestamp": raw["Datetime"].max(),
        "n_rows": len(raw),
    }
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("Saved xgb_model.pkl and PJMW_history.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PJMW_MW_Hourly.xlsx")
# =====================================================================
#  TRAIN THE MODEL AND SAVE IT AS A .PKL FILE  (easy version)
# =====================================================================
#  What this file does, in simple words:
#     1. Reads your Excel file (date + electricity demand)
#     2. Cleans it
#     3. Creates helpful "clues" (features) for the model
#     4. Tests the model on the last 1 year (to measure accuracy)
#     5. Trains the final model on ALL data
#     6. Saves it as  xgb_model.pkl   (the app loads this file)
#
#  How to run:
#     python train_model.py PJMW_MW_Hourly.xlsx
# =====================================================================

import sys
import pickle
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

excel_file = sys.argv[1] if len(sys.argv) > 1 else "PJMW_MW_Hourly.xlsx"


# ---------------------------------------------------------------------
# STEP 1: Read the data
# ---------------------------------------------------------------------
print("Step 1: reading data ...")
df = pd.read_excel(excel_file)                    # columns: Datetime, PJMW_MW
df["Datetime"] = pd.to_datetime(df["Datetime"])   # make sure dates are real dates


# ---------------------------------------------------------------------
# STEP 2: Clean the data
# ---------------------------------------------------------------------
print("Step 2: cleaning data ...")
# Some hours appear twice (daylight-saving clock change).
# We keep one row per hour by taking the average.
df = df.groupby("Datetime", as_index=False)["PJMW_MW"].mean()
df = df.sort_values("Datetime").reset_index(drop=True)   # oldest -> newest

# Save the clean history. The app uses it to start the forecast.
df.to_csv("PJMW_history.csv", index=False)


# ---------------------------------------------------------------------
# STEP 3: Create features (the "clues" the model learns from)
# ---------------------------------------------------------------------
print("Step 3: creating features ...")

# --- 3a. Calendar clues: what time is it? ---
df["hour"]      = df["Datetime"].dt.hour          # 0 - 23
df["Dayofweek"] = df["Datetime"].dt.dayofweek     # Monday=0 ... Sunday=6
df["isweekend"] = (df["Dayofweek"] >= 5).astype(int)   # 1 if Saturday/Sunday
df["month"]     = df["Datetime"].dt.month
df["year"]      = df["Datetime"].dt.year

# Is it a US public holiday? (1 = yes, 0 = no)
holidays = USFederalHolidayCalendar().holidays(
    df["Datetime"].min().normalize(),
    df["Datetime"].max().normalize()
)
df["Is_Holiday"] = df["Datetime"].dt.normalize().isin(holidays).astype(int)

# --- 3b. Past-demand clues: what happened before? ---
# lag_1   = demand 1 hour ago
df["lag_1"] = df["PJMW_MW"].shift(1)

# lag_24  = demand at the same hour yesterday
# lag_168 = demand at the same hour last week (168 hours = 7 days)
# (We match by timestamp, so missing hours never break the logic.)
for hours_back in (24, 168):
    past = df[["Datetime", "PJMW_MW"]].copy()
    past["Datetime"] = past["Datetime"] + pd.Timedelta(hours=hours_back)
    past = past.rename(columns={"PJMW_MW": f"lag_{hours_back}"})
    df = df.merge(past, on="Datetime", how="left")

# rolling averages = average demand of the previous 24 hours / 7 days
df["rolling_24"]  = df["PJMW_MW"].shift(1).rolling(24).mean()
df["rolling_168"] = df["PJMW_MW"].shift(1).rolling(168).mean()

# --- 3c. Circle (sin/cos) clues ---
# Hour 23 and hour 0 are neighbours, but 23 and 0 look far apart as numbers.
# sin/cos turn the clock into a circle so the model understands this.
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]  = np.sin(2 * np.pi * df["Dayofweek"] / 7)
df["dow_cos"]  = np.cos(2 * np.pi * df["Dayofweek"] / 7)

# The first rows have no "past" yet, so they contain empty values. Remove them.
df = df.dropna().reset_index(drop=True)

# The list of clues the model will use (X), and the answer it must predict (y)
FEATURES = [
    "hour", "Dayofweek", "isweekend", "Is_Holiday", "month", "year",
    "lag_1", "lag_24", "lag_168", "rolling_24", "rolling_168",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
TARGET = "PJMW_MW"


# ---------------------------------------------------------------------
# STEP 4: Split into TRAIN (old data) and TEST (last 1 year)
# ---------------------------------------------------------------------
print("Step 4: splitting train / test ...")
test_start = df["Datetime"].max() - pd.DateOffset(years=1)

train = df[df["Datetime"] <  test_start]     # model learns from this
test  = df[df["Datetime"] >= test_start]     # model is exam-tested on this


# ---------------------------------------------------------------------
# STEP 5: Create the model (best settings found in your notebook tuning)
# ---------------------------------------------------------------------
def make_model():
    return XGBRegressor(
        n_estimators=1000,      # number of trees
        learning_rate=0.1,      # how fast it learns
        max_depth=7,            # how deep each tree can go
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------
# STEP 6: Test the accuracy (train on old data, predict the last year)
# ---------------------------------------------------------------------
print("Step 6: testing accuracy (takes a minute) ...")
test_model = make_model()
test_model.fit(train[FEATURES], train[TARGET])
predictions = test_model.predict(test[FEATURES])

mae  = mean_absolute_error(test[TARGET], predictions)                       # average error in MW
rmse = np.sqrt(mean_squared_error(test[TARGET], predictions))               # punishes big errors
mape = mean_absolute_percentage_error(test[TARGET], predictions) * 100      # error in %

print(f"   MAE  = {mae:.1f} MW")
print(f"   RMSE = {rmse:.1f} MW")
print(f"   MAPE = {mape:.2f} %")


# ---------------------------------------------------------------------
# STEP 7: Train the FINAL model on ALL data
# ---------------------------------------------------------------------
print("Step 7: training final model on all data ...")
final_model = make_model()
final_model.fit(df[FEATURES], df[TARGET])


# ---------------------------------------------------------------------
# STEP 8: Save everything into ONE .pkl file
# ---------------------------------------------------------------------
# A pkl file is like a "saved game": the trained model is stored so we
# never have to train again. We also store the feature list and the
# accuracy numbers so the app can show them.
package = {
    "model": final_model,
    "features": FEATURES,
    "metrics": {"MAE": float(mae), "RMSE": float(rmse), "MAPE_%": float(mape)},
}

with open("xgb_model.pkl", "wb") as file:       # "wb" = write binary
    pickle.dump(package, file)

print("Step 8: saving ...")
print("Done!  Saved  xgb_model.pkl  and  PJMW_history.csv")


# =====================================================================
#  BONUS: how to USE the saved pkl file (for your understanding)
# =====================================================================
#   import pickle, pandas as pd
#
#   with open("xgb_model.pkl", "rb") as f:      # "rb" = read binary
#       package = pickle.load(f)
#
#   model    = package["model"]                 # the trained XGBoost
#   features = package["features"]              # the 15 column names
#
#   new_row = pd.DataFrame([{ ... 15 values ... }])
#   prediction = model.predict(new_row[features])
#   print(prediction)                           # demand in MW
# =====================================================================