# frontend/dashboard.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API = "http://127.0.0.1:5000"

st.set_page_config(layout="wide")
st.title("🔍 Insider Threat Detection — Multi-source Dashboard")

# Sidebar Navigation
st.sidebar.title("📑 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Risky Users", "📈 User Features", "🗂️ Raw Logs"]
)

# Refresh model button
st.sidebar.markdown("### Actions")
if st.sidebar.button("🔄 Refresh model (reload CSVs & retrain)"):
    try:
        r = requests.post(f"{API}/refresh")
        r.raise_for_status()
        st.sidebar.success("✅ Model refreshed")
    except:
        st.sidebar.error("❌ Could not connect to backend.")

# ---------------- Fetch risky users ----------------
try:
    r = requests.get(f"{API}/risky_users")
    r.raise_for_status()
    risky_list = r.json()
except Exception as e:
    st.error("❌ Could not reach backend. Make sure backend is running (`python backend/app.py`).")
    st.stop()

# Safe conversion to DataFrame
if isinstance(risky_list, list) and len(risky_list) > 0 and isinstance(risky_list[0], dict):
    risky_df = pd.DataFrame(risky_list)
else:
    risky_df = pd.DataFrame()

# Normalize column names safely
if not risky_df.empty:
    col_mapping = {}
    if 'user' in risky_df.columns: col_mapping['user'] = 'User'
    if 'isolation_forest' in risky_df.columns: col_mapping['isolation_forest'] = 'Mean Risk'
    if 'rank' in risky_df.columns: col_mapping['rank'] = 'Anomaly Rank'
    risky_df = risky_df.rename(columns=col_mapping)

# ---------------------- PAGE 1: HOME ----------------------
if page == "🏠 Home":
    
    

    if not risky_df.empty:
        st.subheader("Top 5 Risky Users (Quick Glance)")
        st.dataframe(risky_df.head(5))

        fig = px.bar(
            risky_df.head(10),
            x="User",
            y="Mean Risk" if "Mean Risk" in risky_df.columns else None,
            color="Anomaly Rank" if "Anomaly Rank" in risky_df.columns else None,
            title="Top 10 Users by Risk Score",
            text="Anomaly Rank" if "Anomaly Rank" in risky_df.columns else None
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- PAGE 2: RISKY USERS ----------------------
elif page == "📊 Risky Users":
    st.header("📊 Top Risky Users")
    if risky_df.empty:
        st.warning("⚠️ No data available yet. Check your CSVs in backend/data/")
    else:
        tabs = st.tabs(["📋 Table", "📊 Risk Scores"])
        with tabs[0]:
            display_cols = [c for c in ['User','Mean Risk','Anomaly Rank'] if c in risky_df.columns]
            st.dataframe(risky_df[display_cols])

        with tabs[1]:
            fig = px.bar(
                risky_df,
                x="User",
                y="Mean Risk" if "Mean Risk" in risky_df.columns else None,
                color="Anomaly Rank" if "Anomaly Rank" in risky_df.columns else None,
                title="Risk Scores by User",
                text="Anomaly Rank" if "Anomaly Rank" in risky_df.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------- PAGE 3: USER FEATURES ----------------------
elif page == "📈 User Features":
    if risky_df.empty:
        st.warning("⚠️ No data available yet.")
    else:
        user = st.selectbox("Select User to inspect", options=risky_df['User'].tolist())
        if user:
            f = requests.get(f"{API}/user/features", params={"user": user}).json()
            if isinstance(f, list) and len(f) > 0 and isinstance(f[0], dict):
                feats = pd.DataFrame(f)
            else:
                feats = pd.DataFrame()

            if not feats.empty:
                if 'date' in feats.columns:
                    feats['date'] = pd.to_datetime(feats['date'])
                    feats = feats.sort_values('date')

                st.subheader(f"User: {user} — Per-day Features")
                st.dataframe(feats)

                plot_cols = [c for c in feats.columns if c not in ('user','date','anomaly')]
                if plot_cols:
                    st.markdown("#### 📈 Time Series of Features")
                    fig = px.line(feats, x="date", y=plot_cols, title="User Features over Time")
                    st.plotly_chart(fig, use_container_width=True)

                if "anomaly" in feats.columns:
                    st.markdown("#### 🚨 Anomaly Timeline")
                    anomaly_days = feats[feats["anomaly"] == 1]
                    if not anomaly_days.empty and 'activity_count' in anomaly_days.columns:
                        fig = px.scatter(
                            anomaly_days, x="date", y="activity_count",
                            color_discrete_sequence=["red"],
                            title="Anomalous Days (Activity Count shown)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("✅ No anomalies detected for this user.")
            else:
                st.info("ℹ️ No features available for this user.")

# ---------------------- PAGE 4: RAW LOGS ----------------------
elif page == "🗂️ Raw Logs":
    if risky_df.empty:
        st.warning("⚠️ No data available yet.")
    else:
        user = st.selectbox("Select User to view logs", options=risky_df['User'].tolist())
        if user:
            raw = requests.get(f"{API}/user/raw", params={"user": user}).json()
            if isinstance(raw, dict):
                st.subheader(f"📜 Raw logs for {user}")
                for src, rows in raw.items():
                    with st.expander(f"{src.upper()} — {len(rows)} rows"):
                        if len(rows) > 0:
                            df = pd.DataFrame(rows)
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                st.dataframe(df.sort_values('timestamp', ascending=False).head(200))
                            else:
                                st.dataframe(df.head(200))
            else:
                st.info("ℹ️ No raw logs available for this user.")