# streamlit_app.py
import streamlit as st
import os
import pandas as pd
from datetime import timedelta

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
def _load_tips_records(artifacts_dir: str):
    import json
    tips = []
    path = os.path.join(artifacts_dir, "chunks_tokenized.jsonl")
    if not os.path.exists(path):
        return tips
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                src = rec.get("source", "")
                # Consider anything in a /tips/ folder or filename containing "tips"
                if "/tips/" in src or os.sep + "tips" + os.sep in src or "tips" in os.path.basename(src).lower():
                    tips.append({
                        "text": rec.get("text", ""),
                        "source": src,
                        "page": rec.get("page"),
                        "filename": os.path.basename(src),
                    })
    except Exception:
        return []
    return tips

def _get_daily_tip(artifacts_dir: str, max_chars: int = 220) -> str:
    tips = _load_tips_records(artifacts_dir)
    if not tips:
        return ""
    from datetime import date
    import random
    seed = date.today().isoformat()
    rng = random.Random(seed)
    rec = rng.choice(tips)
    txt = (rec.get("text") or "").strip().replace("\n", " ")
    if len(txt) > max_chars:
        txt = txt[:max_chars].rstrip() + "…"
    return txt

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
tips = _load_tips_records(artifacts_dir)
if tips:
    from datetime import date
    import random, io
    
    rng = random.Random(date.today().isoformat())
    rec = rng.choice(tips)  # Pick one deterministically for today

    # Prepare the tip text
    tip_text = (rec.get("text") or "").strip().replace("\n", " ")
    if len(tip_text) > max_chars:
        tip_text = tip_text[:max_chars].rstrip() + "…"

    # Show tip and source
    st.info(f"💡 **Daily Tip:** {tip_text}")
    with st.expander("Tip source"):
        src_path = rec.get("source") or ""
        filename = os.path.basename(src_path)
        src_page = rec.get("page", "?")

        # Try a few likely locations for the source file
        candidates = [
            src_path,  # as recorded in metadata
            os.path.join(os.getcwd(), src_path),
            os.path.join("knowledge_base", os.path.basename(src_path)),  
        ]
        file_bytes = None
        for p in candidates:
            if p and os.path.exists(p) and os.path.isfile(p):
                try:
                    with open(p, "rb") as fh:
                        file_bytes = fh.read()
                    break
                except Exception:
                    pass

        if file_bytes:
            st.caption(f"[source: {filename} p.{page}]")
            st.download_button(
                label=f"Download {filename}",
                data=file_bytes,
                file_name=filename,
                mime="application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream",
            )
        else:
            # If we can't read the file, at least show the reference cleanly
            st.caption(f"[source: {filename} p.{src_page}] (file not found for download)")


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

        symptoms = st.selectbox(
            "Main Symptom",
            [
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
            ],
        )
        pain_level = st.slider("Self-Reported Pain Level (1 = None, 10 = Extreme)", 1, 10, 5)
        emotion    = st.selectbox("Self-Reported Emotion", ["Anxious", "Tired", "Neutral", "Sad", "Angry", "Happy"], index=2)

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
            You reported feeling *{emotion}* with a pain level of **{pain_level}**.  
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
                "symptoms": symptoms,
                "pain_level": pain_level,   # actual
                "emotion": emotion,         # actual
                "detected_triggers": ", ".join(triggers_found) if triggers_found else "",
                "rag_insight": rag_insight,
                "rag_sources": " || ".join(source_docs) if show_sources and source_docs else "",
            }
        )
        st.cache_data.clear()
        st.success("Entry saved.")
        
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

            # Compact cards with expanders
            st.caption(f"Showing {len(results)} of {len(logs_df)} entries after filters.")
            for _, row in results.iterrows():
                header = f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')} — {row.get('emotion','?')} | Pain {row.get('pain_level','?')}"
                with st.expander(header):
                    st.markdown(f"**Insight:** {row.get('rag_insight','')}")
                    st.markdown(f"**Notes:** {row.get('free_text_notes','')}")
                    if isinstance(row.get("rag_sources"), str) and row["rag_sources"].strip():
                        st.markdown("**Sources:**")
                        for i, src in enumerate(str(row["rag_sources"]).split(" || ")):
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