from pathlib import Path
import ast
import re
import warnings

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    mean_absolute_error = mean_squared_error = r2_score = MinMaxScaler = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None


PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

if sns is not None:
    sns.set_theme(style="whitegrid", context="notebook")
warnings.filterwarnings("ignore")


def resolve_raw_csv():
    candidates = [
        DATA_DIR / "borg_traces_data.csv",
        PROJECT_ROOT / "borg_traces_data.csv",
        Path("/mnt/data/borg_traces_data.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Place borg_traces_data.csv in the data folder, or make it available at /mnt/data/borg_traces_data.csv."
    )


def load_csv(path, max_rows=None):
    return pd.read_csv(path, nrows=max_rows)


def find_column(df, candidates, required=True):
    normalized = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    for col in df.columns:
        low = col.lower()
        if any(candidate.lower() in low for candidate in candidates):
            return col
    if required:
        raise ValueError(f"Could not find a matching column for: {candidates}. Available columns: {list(df.columns)}")
    return None


def infer_time_unit(values):
    positive = pd.Series(values).dropna()
    positive = positive[positive > 0]
    if positive.empty:
        return "s"
    q95 = positive.quantile(0.95)
    if q95 > 1e15:
        return "ns"
    if q95 > 1e12:
        return "us"
    if q95 > 1e9:
        return "ms"
    return "s"


def extract_cpu_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return float(parsed.get("cpus", np.nan))
    except (ValueError, SyntaxError, TypeError):
        pass
    match = re.search(r"'?cpus'?\s*:\s*([0-9.eE+-]+)", text)
    if match:
        return float(match.group(1))
    return pd.to_numeric(text, errors="coerce")


def reconstruct_timestamp(df, time_col, start="2019-05-01"):
    values = pd.to_numeric(df[time_col], errors="coerce").fillna(0)
    values = values - values.min()
    unit = infer_time_unit(values)
    return pd.to_datetime(start) + pd.to_timedelta(values, unit=unit)


def choose_interval(index, fallback="5min"):
    diffs = pd.Series(index).sort_values().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(fallback)
    return diffs.median()


def clean_cpu_timeseries(df, time_col, cpu_col, start="2019-05-01", interval="5min"):
    working = df[[time_col, cpu_col]].copy()
    working.columns = ["raw_time", "raw_cpu"]
    working["cpu_utilization"] = working["raw_cpu"].map(extract_cpu_value)
    working = working.dropna(subset=["cpu_utilization"])
    working["timestamp"] = reconstruct_timestamp(working, "raw_time", start=start)
    working = working[["timestamp", "cpu_utilization"]].sort_values("timestamp")

    aggregated = working.groupby("timestamp", as_index=False)["cpu_utilization"].mean()
    uniform = (
        aggregated.set_index("timestamp")
        .sort_index()
        .resample(interval)
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reset_index()
    )
    return working, aggregated, uniform


def make_supervised(series, lags=12, horizon=1):
    df = pd.DataFrame({"cpu_utilization": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["cpu_utilization"].shift(lag)
    df["rolling_mean_3"] = df["cpu_utilization"].shift(1).rolling(3).mean()
    df["rolling_mean_6"] = df["cpu_utilization"].shift(1).rolling(6).mean()
    df["rolling_std_6"] = df["cpu_utilization"].shift(1).rolling(6).std()
    df["target"] = df["cpu_utilization"].shift(-horizon)
    df = df.dropna()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, df


def train_test_split_time(df, test_ratio=0.2):
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def save_prediction_csv(name, timestamps, actual, predicted):
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "actual": np.asarray(actual),
        name: np.asarray(predicted),
    })
    path = OUTPUT_DIR / f"{name}_predictions.csv"
    out.to_csv(path, index=False)
    return path, out


def plot_actual_predicted(df, pred_col, title, filename):
    if plt is None:
        raise ImportError("Install matplotlib to create plots.")
    plt.figure(figsize=(15, 6))
    plt.plot(df["timestamp"], df["actual"], label="Actual", linewidth=2)
    plt.plot(df["timestamp"], df[pred_col], label=pred_col.replace("_", " ").title(), linewidth=2)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("CPU Utilization")
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=160)
    plt.show()
    return path


def scale_lstm_data(train_values, test_values):
    if MinMaxScaler is None:
        raise ImportError("Install scikit-learn to scale LSTM data.")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test_values.reshape(-1, 1)).ravel()
    return scaler, train_scaled, test_scaled


def make_lstm_sequences(values, timestamps, window=24):
    X, y, t = [], [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
        t.append(timestamps[i])
    return np.array(X)[..., None], np.array(y), pd.to_datetime(t)
