import streamlit as st
import os
import pandas as pd
import joblib
from datetime import datetime
from model_inference import predict_pain_emotion
from visualizations import (
    plot_pain_trend, 
    plot_emotion_dist, 
    plot_pain_by_emotion, 
    plot_pain_emotion_heatmap
)

# # Ensure the 'data' directory exists
# os.makedirs("data", exist_ok=True)

# Load historical logs if any (you can save this as a CSV)
@st.cache_data
def load_logs():
    try:
        return pd.read_csv("data/user_logs.csv", parse_dates=["timestamp"])
    except FileNotFoundError:
        return pd.DataFrame(columns=["timestamp", "free_text_notes", "age", 
                                     "pain_location", "gender", "symptoms",
                                     "pain_level", "predicted_pain_level",
                                     "emotion", "predicted_emotion"
])

# Save a new log
def save_log(entry):
    logs = load_logs()
    logs = pd.concat([logs, pd.DataFrame([entry])], ignore_index=True)
    logs.to_csv("data/user_logs.csv", index=False)

# Sidebar
st.sidebar.title("Smart Sickle Cell Diary")
page = st.sidebar.radio("Navigate", ["📖 Log Entry", "📊 Dashboard", "📁 View Past Logs"])

# --- Log Entry Page ---
if page == "📖 Log Entry":
    st.title("Log a New Diary Entry")

    with st.form("diary_form"):
        free_text_notes = st.text_area("How are you feeling today?", height=150)
        age = st.slider("Age", min_value=1, max_value=100, value=25)
        pain_location = st.selectbox("Pain Location", ["Back", "Head", "Chest", "Abdomen", "Legs"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        symptoms = st.selectbox("Main Symptom", ['muscle cramps and joint pain',
                                                 'pain in the chest and shortness of breath',
                                                 'frequent episodes of joint pain', 'occasional chest pain',
                                                 'headaches and dizziness', 'chronic abdominal pain',
                                                 'constant headache', 'joint pain and tiredness',
                                                 'pain in limbs and joints', 'tiredness and body aches',
                                                 'fatigue and back pain', 'painful swelling in joints and fatigue', 
                                                 'chronic pain and fatigue', 
                                                 'frequent back pain', 'chronic pain and nausea',
                                                 'painful episodes and anemia', 'headaches and anxiety',
                                                 'painful episodes triggered by stress', 'anemia and leg pain'])
        pain_level = st.slider("Self-Reported Pain Level (1 = None, 10 = Extreme)", 1, 10, 5)
        emotion = st.selectbox(
            "Self-Reported Emotion",
            options=["Anxious", "Tired", "Neutral", "Sad", "Angry", "Happy"],
            index=2
        )
        submitted = st.form_submit_button("Submit")

    if submitted:
        user_input = {
            "free_text_notes": free_text_notes,
            "age": age,
            "pain_location": pain_location,
            "gender": gender,
            "symptoms": symptoms,
            "emotion": emotion  
        }

        predicted_pain_level, predicted_emotion = predict_pain_emotion(user_input)
        
        st.success(f"Predicted Pain Level: {round(predicted_pain_level)}")
        st.markdown(f"**You Reported:** {emotion} | **Model Suggests:** {predicted_emotion}")
        st.markdown(f"Based on your notes, the system noticed patterns similar to people reporting *{predicted_emotion}* or *Moderate–High Pain*.")

        with st.expander("What does this mean?"):
            st.markdown("""
            These outputs are **suggestions based on patterns in your input**.  
            They are not medical diagnoses and should not replace your lived experience or clinical judgment.  
            Consider discussing these entries with your healthcare provider.
            """)

        log_entry = {
            "timestamp": pd.Timestamp.now(),
            "free_text_notes": free_text_notes,
            "age": age,
            "pain_location": pain_location,
            "gender": gender,
            "symptoms": symptoms,
            "pain_level": pain_level,
            "predicted_pain": round(predicted_pain_level),
            "emotion": emotion,
            "predicted_emotion": predicted_emotion
        }
        save_log(log_entry)
        st.cache_data.clear()  # Clear the cached logs
        # st.rerun()

# --- Dashboard Page ---
elif page == "📊 Dashboard":
    st.title("Diary Dashboard")
    logs_df = load_logs()

    if logs_df.empty:
        st.info("No diary entries available yet. Log your first entry to get started!")
    else:
        st.subheader("Recent Entries")
        st.dataframe(logs_df.sort_values("timestamp", ascending=False).head(5))

        st.subheader("Pain Level Trend")
        st.plotly_chart(plot_pain_trend(logs_df), use_container_width=True)

        st.subheader("Emotion Distribution")
        st.plotly_chart(plot_emotion_dist(logs_df), use_container_width=True)
        
        st.subheader("Average Pain by Emotion")
        st.plotly_chart(plot_pain_by_emotion(logs_df), use_container_width=True)

        st.subheader("Pain Location vs Emotion")
        st.plotly_chart(plot_pain_emotion_heatmap(logs_df), use_container_width=True)

# --- View Past Logs Page ---
elif page == "📁 View Past Logs":
    st.title("View Past Logs")
    logs_df = load_logs()

    if logs_df.empty:
        st.info("No logs to show yet.")
    else:
        st.subheader("Filters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            start_date = st.date_input("Start Date", value=logs_df["timestamp"].min().date())
        with col2:
            end_date = st.date_input("End Date", value=logs_df["timestamp"].max().date())
        with col3:
            selected_emotion = st.multiselect("Emotion", options=logs_df["predicted_emotion"].unique())
        with col4:
            selected_pain_level = st.multiselect("Pain Level", options=logs_df["predicted_pain"].unique())

        # Apply filters
        mask = (logs_df["timestamp"].dt.date >= start_date) & (logs_df["timestamp"].dt.date <= end_date)
        if selected_emotion:
            mask &= logs_df["predicted_emotion"].isin(selected_emotion)
        if selected_pain_level:
            mask &= logs_df["predicted_pain"].isin(selected_pain_level)
            
        filtered_logs = logs_df[mask]
        st.dataframe(filtered_logs.sort_values("timestamp", ascending=False), use_container_width=True)
        
        st.caption(f"🔍 Showing {len(filtered_logs)} of {len(logs_df)} total entries after filtering.")

        csv = filtered_logs.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "filtered_logs.csv", "text/csv", key='download-csv')
