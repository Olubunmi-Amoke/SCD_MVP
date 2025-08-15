import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re

# --- Helpers ---
def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def _nonempty(df: pd.DataFrame, cols) -> bool:
    return all(c in df.columns for c in cols) and not df.empty

# Pain Trend Line Chart
# Purpose: Displays how a user's reported pain levels fluctuate over time; ; falls back gracefully if timestamps missing.
# Value: Helps users recognize patterns in pain episodes (e.g., recurring spikes), potentially linked to medication, stress, or activities.
def plot_pain_trend(df: pd.DataFrame):
    if not _nonempty(df, ["timestamp", "pain_level"]):
        return go.Figure()
    df = _ensure_timestamp(df).dropna(subset=["timestamp", "pain_level"])
    if df.empty:
        return go.Figure()

    df_sorted = df.sort_values("timestamp")
    # Optional daily mean smoothing 
    daily = df_sorted.set_index("timestamp").resample("D")["pain_level"].mean().reset_index()

    fig = px.line(daily, x="timestamp", y="pain_level", title="Pain Level Over Time")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(xaxis_title="Date", yaxis_title="Pain Level (1–10)")
    return fig


# Emotion Distribution Histogram
# Purpose: Shows the frequency of each reported emotion class in the dataset.
# Value: Provides a high-level overview of emotional patterns, useful for spotting emotional states that dominate pain entries.
def plot_emotion_dist(df: pd.DataFrame):
    if not _nonempty(df, ["emotion"]):
        return go.Figure()
    d = df.copy()
    d = d.dropna(subset=["emotion"])
    if d.empty:
        return go.Figure()

    # Keep order by frequency
    order = d["emotion"].value_counts().index.tolist()
    fig = px.histogram(d, x="emotion", category_orders={"emotion": order}, title="Emotion Frequency")
    fig.update_layout(xaxis_title="Emotion", yaxis_title="Count")
    return fig


# Average Pain by Emotion Bar Chart
# Purpose: Displays the average pain level for each emotion class.
# Value: Highlights relationships between emotions and pain levels (e.g., higher pain with "angry" or "scared"), guiding emotional awareness and coping.
def plot_pain_by_emotion(df: pd.DataFrame):
    if not _nonempty(df, ["emotion", "pain_level"]):
        return go.Figure()
    d = df.dropna(subset=["emotion", "pain_level"])
    if d.empty:
        return go.Figure()

    avg = d.groupby("emotion", dropna=True)["pain_level"].mean().reset_index()
    avg = avg.sort_values("pain_level", ascending=False)
    fig = px.bar(avg, x="emotion", y="pain_level", title="Average Pain Level by Emotion")
    fig.update_layout(xaxis_title="Emotion", yaxis_title="Avg Pain (1–10)")
    return fig

# Pain Location vs Emotion Heatmap
# Purpose: Creates a heatmap showing co-occurrence between pain locations and emotions.
# Value: Helps users in recognizing how certain pain areas correlate with specific emotional states (adds spatial-emotional insight).
def plot_pain_emotion_heatmap(df: pd.DataFrame):
    if not _nonempty(df, ["pain_location", "emotion"]):
        return go.Figure()
    d = df.dropna(subset=["pain_location", "emotion"])
    if d.empty:
        return go.Figure()

    # Order by frequency for nicer axes
    emo_order = d["emotion"].value_counts().index.tolist()
    loc_order = d["pain_location"].value_counts().index.tolist()

    heatmap_data = pd.crosstab(d["pain_location"], d["emotion"]).reindex(index=loc_order, columns=emo_order, fill_value=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="Reds",
        )
    )
    fig.update_layout(
        title="Pain Location vs. Emotion Heatmap",
        xaxis_title="Emotion",
        yaxis_title="Pain Location"
    )
    return fig


# Trigger Trends Bar Chart
# Purpose: Shows frequency of detected triggers across user logs.
# Value: Helps users identify recurring patterns that may need attention.
def plot_trigger_trends(df: pd.DataFrame):
    if "detected_triggers" not in df.columns or df["detected_triggers"].isna().all():
        return go.Figure()

    # Split multiple triggers per log and flatten into one list
    # Handles ",", ";", "|", "||", and extra spaces
    parts = (
        df["detected_triggers"]
        .dropna()
        .astype(str)
        .apply(lambda s: re.split(r"[|;,]+|\s\|\|\s|,\s+", s))
    )
    flat = [t.strip() for sub in parts for t in sub if t and t.strip()]

    if not flat:
        return go.Figure()

    # Normalize a bit (lowercase for grouping, then title-case for display)
    norm = pd.Series([t.lower() for t in flat]).value_counts().reset_index()
    norm.columns = ["trigger_norm", "Count"]
    norm["Trigger"] = norm["trigger_norm"].str.title()

    fig = px.bar(norm, x="Trigger", y="Count", title="Most Frequent Triggers")
    fig.update_layout(xaxis_tickangle=-30)
    return fig