# streamlit_app.py
import streamlit as st
import os
import pandas as pd
import random
from datetime import timedelta, date

from trigger_keywords import detect_triggers
from visualizations import (
    plot_pain_trend, 
    plot_emotion_dist, 
    plot_pain_by_emotion, 
    plot_pain_emotion_heatmap,
    plot_trigger_trends
)

from rag_integration import get_rag_helper

# Ensure the 'data' directory exists
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="Smart Sickle Cell Diary", page_icon="🩸", layout="wide")

# --- Daily Tip helpers ---
@st.cache_data
def _get_fixed_tips():
    return [
        # Hydration & Nutrition
        "Stay hydrated: Drink at least 8 cups of water daily.",
        "Eat iron-rich foods like spinach, beans, and lean meats.",
        "Include vitamin-rich fruits and vegetables in your meals.",
        "Limit processed foods and sugary drinks.",

        # Lifestyle & Activity
        "Balance rest with light exercise to maintain circulation.",
        "Practice deep breathing or relaxation exercises daily.",
        "Avoid extreme temperatures to reduce pain crises.",
        "Dress appropriately for weather changes to prevent cold-related pain.",

        # Symptom Tracking & Medical Care
        "Track your symptoms daily to notice patterns early.",
        "Take medications exactly as prescribed.",
        "Schedule and attend regular medical check-ups.",
        "Keep an emergency plan ready for pain crises.",

        # Emotional & Social Wellbeing
        "Share how you’re feeling with trusted friends or family.",
        "Join a support group for people living with SCD.",
        "Engage in hobbies that bring you joy and relaxation.",
        "Practice gratitude by writing down one positive thing each day."
    ]

def _get_daily_tip() -> str:
    tips = _get_fixed_tips()
    if not tips:
        return ""
    rng = random.Random(date.today().isoformat())
    return rng.choice(tips)

# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data
def load_logs():
    try:
        df = pd.read_csv("data/user_logs.csv", parse_dates=["timestamp"])
        return df
    except FileNotFoundError:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "free_text_notes",
                "age",
                "pain_location",
                "gender",
                "symptoms",
                "pain_level",           # actual, user-entered
                "emotion",              # actual, user-entered
                "detected_triggers",
                "rag_insight",
                "rag_sources",
            ]
        )

def save_log(entry: dict):
    logs = load_logs()
    logs = pd.concat([logs, pd.DataFrame([entry])], ignore_index=True)
    logs.to_csv("data/user_logs.csv", index=False)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Smart Sickle Cell Diary")
page = st.sidebar.radio("Navigate", ["📖 Log Entry", "📊 Dashboard"], key="nav_page")
st.caption(f"🧭 debug: page = {page!r}")

st.sidebar.markdown("---")
st.sidebar.subheader("RAG Settings")
artifacts_dir   = st.sidebar.text_input("Artifacts directory", value="artifacts")
k_top           = st.sidebar.slider("Top-K contexts", 1, 10, 3)
llm_model       = st.sidebar.text_input("LLM model", value="gpt-4o-mini")
use_llm         = st.sidebar.checkbox("Use LLM (cost $$$)", value=False)
show_sources    = st.sidebar.checkbox("Show sources", value=True)
max_chars       = st.sidebar.slider("Max chars per context", 400, 2000, 1200, 100)
st.sidebar.caption("Set OPENAI_API_KEY to enable generation.")

# Instantiate RAG helper lazily (cached inside get_rag_helper)
rag = get_rag_helper(artifacts_dir=artifacts_dir)

# --- Daily Tip banner (top of app) ---
tip_text = _get_daily_tip()
if tip_text:
    st.info(f"💡 **Daily Tip:** {tip_text}")
    if use_llm:
        with st.expander("Tip source"):
            st.caption("This tip is from a fixed wellness tips list curated for SCD self-care.")

# -----------------------------
# Log Entry Page
# -----------------------------
if page == "📖 Log Entry":
    st.title("Log a New Diary Entry")

    with st.form("diary_form"):
        free_text_notes = st.text_area("How are you feeling today?", height=150)
        colA, colB, colC = st.columns(3)
        with colA:
            age = st.slider("Age", min_value=1, max_value=100, value=25)
        with colB:
            pain_location = st.selectbox("Pain Location", ["Back", "Head", "Chest", "Abdomen", "Legs"])
        with colC:
            gender = st.selectbox("Gender", ["Male", "Female"])  # extend as needed

        # --- Multi-select for symptoms with 'Other' option ---
        default_symptoms = [
                "muscle cramps and joint pain",
                "pain in the chest and shortness of breath",
                "frequent episodes of joint pain",
                "occasional chest pain",
                "headaches and dizziness",
                "chronic abdominal pain",
                "constant headache",
                "joint pain and tiredness",
                "pain in limbs and joints",
                "tiredness and body aches",
                "fatigue and back pain",
                "painful swelling in joints and fatigue",
                "chronic pain and fatigue",
                "frequent back pain",
                "chronic pain and nausea",
                "painful episodes and anemia",
                "headaches and anxiety",
                "painful episodes triggered by stress",
                "anemia and leg pain",
                "Other"
            ],
        
        if "symptom_options" not in st.session_state:
            st.session_state.symptom_options = default_symptoms.copy()

        selected_symptoms = st.multiselect(
            "Select Symptoms",
            options=st.session_state.symptom_options,
        )

        if "Other" in selected_symptoms:
            new_symptom = st.text_input("Add a new symptom (if not listed)")
            if new_symptom and new_symptom not in st.session_state.symptom_options:
                st.session_state.symptom_options.insert(-1, new_symptom)
                st.success(f"Added new symptom: {new_symptom}")
                
        pain_level = st.slider("Self-Reported Pain Level (1 = None, 10 = Extreme)", 1, 10, 5)
        # --- Multi-select for emotions with 'Other' option ---
        default_emotions = ["Anxious", "Tired", "Neutral", "Sad", "Angry", "Happy", "Other"]
        if "emotion_options" not in st.session_state:
            st.session_state.emotion_options = default_emotions.copy()

        selected_emotions = st.multiselect(
            "Select Emotions",
            options=st.session_state.emotion_options,
            default=["Neutral"]
        )

        if "Other" in selected_emotions:
            new_emotion = st.text_input("Add a new emotion (if not listed)")
            if new_emotion and new_emotion not in st.session_state.emotion_options:
                st.session_state.emotion_options.insert(-1, new_emotion)
                st.success(f"Added new emotion: {new_emotion}")
                
        submitted = st.form_submit_button("Submit")
        
    if submitted:
        # Trigger detection
        triggers_found = detect_triggers(free_text_notes) or []

        # RAG suggestion (helper handles cost controls)
        with st.spinner("Generating supportive suggestions..."):
            rag_insight, source_docs = rag.suggest(
                free_text_notes,
                k=k_top,
                model=llm_model,
                use_llm=use_llm,
                show_sources=show_sources,
                max_chunk_chars=max_chars,
            )

        # Insight banner & context
        st.markdown(
            f"""
            🧠 **Helpful Insight Based on Your Log**  
            You reported feeling *{', '.join([e for e in selected_emotions if e != 'Other'])}* with a pain level of **{pain_level}**.  
            """
        )

        if triggers_found:
            st.markdown(f"🔍 **Possible Triggers Detected:** {', '.join(triggers_found).title()}")
            st.markdown("💡 You might consider reviewing these patterns over time or discussing them with a provider.")

        # --- Transparent RAG output ---
        st.markdown("🧾 **Supportive Suggestion Based on Your Entry:**")
        st.success(rag_insight)

        if show_sources and source_docs:
            with st.expander("🧩 Sources Used"):
                for i, src in enumerate(source_docs):
                    clipped = src[:max_chars] + ("..." if len(src) > max_chars else "")
                    st.markdown(f"{i+1}. {clipped}")

        # Persist log (no predicted fields)
        save_log(
            {
                "timestamp": pd.Timestamp.now(),
                "free_text_notes": free_text_notes,
                "age": age,
                "pain_location": pain_location,
                "gender": gender,
                "symptoms": ", ".join([s for s in selected_symptoms if s != "Other"]),
                "pain_level": pain_level,   # actual
                "emotion": ", ".join([e for e in selected_emotions if e != "Other"]),         # actual
                "detected_triggers": ", ".join(triggers_found) if triggers_found else "",
                "rag_insight": rag_insight,
                "rag_sources": " || ".join(source_docs) if show_sources and source_docs else "",
            }
        )
        st.cache_data.clear()
        st.success("Entry saved.")
        
        if use_llm:
            with st.expander("What does this mean?"):
                st.markdown(
                    """
                    These outputs are **reflective suggestions based on common patterns in similar logs**.  
                    They are **not medical advice** and should **not replace your own judgment or your healthcare provider’s input**.  
                    Sharing these results during your next visit may help guide deeper conversation.
                    """
                )
            st.info("This is your record. You decide what matters most.")


# -----------------------------
# Dashboard Page (with Insight Archive)
# -----------------------------
elif page == "📊 Dashboard":
    st.title("Diary Dashboard")
    logs_df = load_logs()

    if logs_df.empty:
        st.info("No diary entries available yet.")
    else:
        tab_overview, tab_archive = st.tabs(["📈 Overview", "🧠 Insight Archive"])

        with tab_overview:
            st.subheader("Recent Entries")
            st.dataframe(logs_df.sort_values("timestamp", ascending=False).head(5), use_container_width=True)

            st.subheader("Pain Level Trend")
            st.plotly_chart(plot_pain_trend(logs_df), use_container_width=True)

            st.subheader("Emotion Distribution")
            st.plotly_chart(plot_emotion_dist(logs_df), use_container_width=True)

            st.subheader("Average Pain by Emotion")
            st.plotly_chart(plot_pain_by_emotion(logs_df), use_container_width=True)

            st.subheader("Pain Location vs Emotion")
            st.plotly_chart(plot_pain_emotion_heatmap(logs_df), use_container_width=True)

            if "detected_triggers" in logs_df.columns and logs_df["detected_triggers"].astype(str).str.len().gt(0).any():
                st.subheader("Trigger Trends")
                timeframe = st.radio("Select Timeframe:", ["All", "Last 30 Days", "Last 7 Days"], horizontal=True)
                now = pd.Timestamp.now()
                df_time = logs_df.copy()
                if timeframe == "Last 7 Days":
                    df_time = df_time[df_time["timestamp"] >= now - timedelta(days=7)]
                elif timeframe == "Last 30 Days":
                    df_time = df_time[df_time["timestamp"] >= now - timedelta(days=30)]
                st.plotly_chart(plot_trigger_trends(df_time), use_container_width=True)

        with tab_archive:
            st.subheader("Search Your Insights")

            # --- Search & Filters  ---
            keyword = st.text_input("Search by keyword (insights, sources, or notes)")
            ts_min = logs_df["timestamp"].min() if not logs_df["timestamp"].isna().all() else pd.Timestamp.now()
            ts_max = logs_df["timestamp"].max() if not logs_df["timestamp"].isna().all() else pd.Timestamp.now()

            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                start_date = st.date_input("Start Date", value=ts_min.date())
            with c2:
                end_date = st.date_input("End Date", value=ts_max.date())
            with c3:
                emos = logs_df.get("emotion", pd.Series(dtype=str)).dropna().unique().tolist()
                emo_sel = st.multiselect("Emotion", options=sorted(emos) if emos else [])
            with c4:
                pains = logs_df.get("pain_level", pd.Series(dtype=float)).dropna().unique().tolist()
                pains_sorted = sorted({int(p) for p in pains}) if pains else []
                pain_sel = st.multiselect("Pain Level", options=pains_sorted)

            mask = (logs_df["timestamp"].dt.date >= start_date) & (logs_df["timestamp"].dt.date <= end_date)

            if keyword:
                mask &= (
                    logs_df["rag_insight"].fillna("").str.contains(keyword, case=False)
                    | logs_df["rag_sources"].fillna("").str.contains(keyword, case=False)
                    | logs_df["free_text_notes"].fillna("").str.contains(keyword, case=False)
                )

            if emo_sel:
                mask &= logs_df.get("emotion").isin(emo_sel)
            if pain_sel:
                mask &= logs_df.get("pain_level").isin(pain_sel)

            results = logs_df[mask].sort_values("timestamp", ascending=False)
            st.caption(f"Showing {len(results)} of {len(logs_df)} entries after filters.")
            
            # ---------------- Quick Clear Filters Button ----------------
            if st.button("Clear Filters"):
                st.experimental_rerun()
            
            # ---------------- Paginated Table + Single Detail Pane ----------------
            results_reset = results.reset_index(drop=True).copy()
            results_reset["id"] = results_reset.index + 1

            # Table preview columns
            preview = results_reset.copy()
            preview["notes_preview"] = preview["free_text_notes"].fillna("").str.slice(0, 120).apply(lambda s: s + ("…" if len(s) == 120 else ""))
            table_cols = ["id", "timestamp", "emotion", "pain_level", "notes_preview"]

            # Pagination controls
            colps, colpn = st.columns([1, 2])
            with colps:
                page_size = st.selectbox("Per page", [5, 10, 20, 50], index=0)
            total_pages = max(1, (len(preview) + page_size - 1) // page_size)
            with colpn:
                page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

            start = (page_num - 1) * page_size
            end = start + page_size
            page_df = preview.iloc[start:end]

            # Render table
            st.dataframe(
                page_df[table_cols].rename(columns={
                    "id": "ID",
                    "timestamp": "Timestamp",
                    "emotion": "Emotion",
                    "pain_level": "Pain",
                    "notes_preview": "Notes (preview)"
                }),
                use_container_width=True,
            )

            # Row picker for details within current page
            def _label_for_row(row):
                ts = row["timestamp"].strftime("%Y-%m-%d %H:%M") if pd.notnull(row["timestamp"]) else "?"
                emo = row.get("emotion", "?")
                pain = row.get("pain_level", "?")
                return f"{int(row['id'])} — {ts} | {emo} | Pain {pain}"

            if not page_df.empty:
                options = page_df.apply(_label_for_row, axis=1).tolist()
                sel_label = st.selectbox("Select an entry to view details", options)
                sel_id = int(sel_label.split(" — ")[0])
                detail_row = results_reset.loc[results_reset["id"] == sel_id].iloc[0]

                with st.expander("Entry Details", expanded=True):
                    st.markdown(f"**Timestamp:** {detail_row['timestamp']}")
                    st.markdown(f"**Emotion:** {detail_row.get('emotion','')}")
                    st.markdown(f"**Pain Level:** {detail_row.get('pain_level','')}")
                    st.markdown(f"**Symptoms:** {detail_row.get('symptoms','')}")
                    st.markdown(f"**Possible Triggers:** {detail_row.get('detected_triggers','')}")
                    st.markdown(f"**Insight:** {detail_row.get('rag_insight','')}")
                    st.markdown(f"**Notes:** {detail_row.get('free_text_notes','')}")

                    if use_llm and isinstance(detail_row.get("rag_sources"), str) and detail_row["rag_sources"].strip():
                        st.markdown("**Sources:**")
                        for i, src in enumerate(str(detail_row["rag_sources"]).split(" || ")):
                            clipped = src[:max_chars] + ("…" if len(src) > max_chars else "")
                            st.markdown(f"{i+1}. {clipped}")

            # Export filtered results
            export_cols = ["timestamp", "emotion", "pain_level", "free_text_notes", "rag_insight", "rag_sources", "detected_triggers", "age", "pain_location", "gender", "symptoms"]
            export_df = results[[c for c in export_cols if c in results.columns]]
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Archive CSV", csv, "insight_archive.csv", "text/csv")

            # Raw logs view (replaces separate 'View Past Logs' page)
            with st.expander("📁 Show Raw Logs (All Columns)"):
                raw_sorted = logs_df.sort_values("timestamp", ascending=False)
                st.dataframe(raw_sorted, use_container_width=True)
                full_csv = raw_sorted.to_csv(index=False).encode("utf-8")
                st.download_button("Download Full Logs CSV", full_csv, "all_logs.csv", "text/csv", key="download-all-logs")