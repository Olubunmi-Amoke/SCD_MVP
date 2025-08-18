import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional

# ---------- helpers ----------

def _cutoff_for_window(window: str) -> datetime:
    now = datetime.now()
    if isinstance(window, str):
        w = window.strip().lower()
        if w.endswith("d"):
            return now - timedelta(days=int(w[:-1]))
        if w.endswith("w"):
            return now - timedelta(weeks=int(w[:-1]))
        if w.endswith("m"):
            # approximate months as 30 days
            return now - timedelta(days=30 * int(w[:-1]))
    # default 30 days
    return now - timedelta(days=30)


def _split_multi(s: Optional[str]) -> list:
    """Split comma/pipe/semicolon-separated strings into a list."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    text = str(s).strip()
    if not text:
        return []
    # normalize common separators
    for sep in ["||", "|", ";", ","]:
        text = text.replace(sep, ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts


def _ensure_datetime(df: pd.DataFrame, col="timestamp") -> pd.DataFrame:
    out = df.copy()
    if col in out and not pd.api.types.is_datetime64_any_dtype(out[col]):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out

# Pain Trend Line Chart
# Purpose: Displays how a user's reported pain levels fluctuate over time; ; falls back gracefully if timestamps missing.
# Value: Helps users recognize patterns in pain episodes (e.g., recurring spikes), potentially linked to medication, stress, or activities.
def plot_pain_trend(df: pd.DataFrame, window: str = "30d"):
    """Line chart of pain_level over time (filtered to window)."""
    if df is None or df.empty or "pain_level" not in df.columns:
        return go.Figure()
    df = _ensure_datetime(df)
    cutoff = _cutoff_for_window(window)
    d = df[df["timestamp"] >= cutoff].sort_values("timestamp").dropna(subset=["timestamp", "pain_level"])
    if d.empty:
        return go.Figure()
    # daily smoothing (mean per day)
    daily = d.set_index("timestamp").resample("D")["pain_level"].mean().reset_index()
    fig = px.line(daily, x="timestamp", y="pain_level", title=f"Pain Level Over Time (last {window})")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(xaxis_title="Date", yaxis_title="Pain Level (1–10)")
    return fig

# Emotion Distribution Histogram
# Purpose: Shows the frequency of each reported emotion class in the dataset.
# Value: Provides a high-level overview of emotional patterns, useful for spotting emotional states that dominate pain entries.
def plot_emotion_dist(df: pd.DataFrame, window: str = "30d"):
    """Histogram of emotions; supports multi-emotion strings via explode."""
    if df is None or df.empty or "emotion" not in df.columns:
        return go.Figure()
    df = _ensure_datetime(df)
    cutoff = _cutoff_for_window(window)
    d = df[df["timestamp"] >= cutoff].copy()
    d["emotion_list"] = d["emotion"].apply(_split_multi)
    d = d.explode("emotion_list")
    d = d.dropna(subset=["emotion_list"])
    if d.empty:
        return go.Figure()
    order = d["emotion_list"].value_counts().index.tolist()
    fig = px.histogram(d, x="emotion_list", category_orders={"emotion_list": order}, title=f"Emotion Distribution (last {window})")
    fig.update_layout(xaxis_title="Emotion", yaxis_title="Count")
    return fig


# Average Pain by Emotion Bar Chart
# Purpose: Displays the average pain level for each emotion class.
# Value: Highlights relationships between emotions and pain levels (e.g., higher pain with "angry" or "scared"), guiding emotional awareness and coping.
def plot_pain_by_emotion(df: pd.DataFrame, window: str = "30d"):
    """Bar chart of average pain by emotion; supports multi-emotion cells."""
    if df is None or df.empty or "emotion" not in df.columns or "pain_level" not in df.columns:
        return go.Figure()
    df = _ensure_datetime(df)
    cutoff = _cutoff_for_window(window)
    d = df[df["timestamp"] >= cutoff].copy()
    d["emotion_list"] = d["emotion"].apply(_split_multi)
    d = d.explode("emotion_list").dropna(subset=["emotion_list", "pain_level"])  # keep valid pairs
    if d.empty:
        return go.Figure()
    grp = d.groupby("emotion_list", as_index=False)["pain_level"].mean().sort_values("pain_level", ascending=False)
    fig = px.bar(grp, x="emotion_list", y="pain_level", title=f"Average Pain by Emotion (last {window})")
    fig.update_layout(xaxis_title="Emotion", yaxis_title="Avg Pain (1–10)")
    return fig

# Pain Location vs Emotion Heatmap
# Purpose: Creates a heatmap showing co-occurrence between pain locations and emotions.
# Value: Helps users in recognizing how certain pain areas correlate with specific emotional states (adds spatial-emotional insight).
def plot_pain_emotion_heatmap(df: pd.DataFrame, window: str = "30d"):
    """Heatmap of average pain per (emotion, date). Supports multi-emotion cells."""
    if df is None or df.empty or "emotion" not in df.columns or "pain_level" not in df.columns:
        return go.Figure()
    df = _ensure_datetime(df)
    cutoff = _cutoff_for_window(window)
    d = df[df["timestamp"] >= cutoff].copy()
    d["emotion_list"] = d["emotion"].apply(_split_multi)
    d = d.explode("emotion_list").dropna(subset=["emotion_list"])  # emotion per row
    if d.empty:
        return go.Figure()
    d["date"] = d["timestamp"].dt.date
    grp = d.groupby(["emotion_list", "date"], as_index=False)["pain_level"].mean()
    pivot = grp.pivot(index="emotion_list", columns="date", values="pain_level").fillna(0)
    fig = px.imshow(pivot,
                    labels=dict(x="Date", y="Emotion", color="Avg Pain"),
                    title=f"Pain vs. Emotion (avg by day) — last {window}")
    return fig

# Trigger Trends Bar Chart
# Purpose: Shows frequency of detected triggers across user logs.
# Value: Helps users identify recurring patterns that may need attention.
def plot_trigger_trends(df: pd.DataFrame, window: str = "30d"):
    """
    Line chart of trigger counts per day, supporting string lists in 'detected_triggers'.
    Expects a 'timestamp' column.
    """
    if df is None or df.empty or "detected_triggers" not in df.columns:
        return go.Figure()
    df = _ensure_datetime(df)
    cutoff = _cutoff_for_window(window)
    d = df[df["timestamp"] >= cutoff].copy()
    d["trig_list"] = d["detected_triggers"].apply(_split_multi)
    d = d.explode("trig_list").dropna(subset=["trig_list"])  # each trigger per row
    if d.empty:
        return go.Figure()
    d["date"] = d["timestamp"].dt.date
    counts = d.groupby(["date", "trig_list"]).size().reset_index(name="count")
    fig = px.line(counts, x="date", y="count", color="trig_list", markers=True,
                  title=f"Trigger Trends (last {window})")
    fig.update_layout(xaxis_title="Date", yaxis_title="Count")
    return fig