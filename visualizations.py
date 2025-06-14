import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Pain Trend Line Chart
# Purpose: Displays how a user's reported pain levels fluctuate over time.
# Value: Helps users recognize patterns in pain episodes (e.g., recurring spikes), potentially linked to medication, stress, or activities.
def plot_pain_trend(df):
    df_sorted = df.sort_values("timestamp")
    return px.line(df_sorted, x="timestamp", y="pain_level", title="Pain Level Over Time")


# Emotion Distribution Histogram
# Purpose: Shows the frequency of each reported emotion class in the dataset.
# Value: Provides a high-level overview of emotional patterns, useful for spotting emotional states that dominate pain entries.
def plot_emotion_dist(df):
    return px.histogram(df, x="emotion", title="Emotion Frequency")


# Pain by Emotion Bar Chart
# Purpose: Displays the average pain level for each emotion class.
# Value: Highlights relationships between emotions and pain levels (e.g., higher pain with "angry" or "scared"), guiding emotional awareness and coping.
def plot_pain_by_emotion(df):
    avg_pain_by_emotion = df.groupby("emotion")["pain_level"].mean().reset_index()
    return px.bar(avg_pain_by_emotion, x="emotion", y="pain_level", title="Average Pain Level by Emotion")


# Pain Location vs Emotion Heatmap
# Purpose: Creates a heatmap showing co-occurrence between pain locations and emotions.
# Value: Helps users in recognizing how certain pain areas correlate with specific emotional states (adds spatial-emotional insight).
def plot_pain_emotion_heatmap(df):
    heatmap_data = pd.crosstab(df['pain_location'], df['emotion'])
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Reds'))
    fig.update_layout(title="Pain Location vs. Emotion Heatmap", xaxis_title="Emotion", yaxis_title="Pain Location")
    return fig
