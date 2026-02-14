%%writefile app.py
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.io as pio
import datetime
import base64
import plotly.express as px
import numpy as np
# ------------------ CONFIGURATION ------------------
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FitPulse Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ PREMIUM CSS STYLING (FULL LIGHT MODE FIX) ------------------
st.markdown("""
<style>
    /* =========================================
       1. GLOBAL VARIABLES & FONTS
       ========================================= */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    :root {
        --light-sky: #EAF6FF;
        --blush-pink: #FFE4EC;
        --frost-blue: #F2FAFF;
        --vanilla: #FFF9E6;
        --soft-cyan: #E0F7FA;
        --sage-light: #ECF4F1;
        --pastel-green: #E6F9EC;
        --alert-red: #FFF1F2;
        --text-dark: #111827;  /* Dark Charcoal */
        --text-grey: #4b5563;
    }

    body, .stApp {
        background-color: #ffffff;
        color: var(--text-dark);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, span, li, div, label {
        color: var(--text-dark) !important;
        font-family: 'Inter', sans-serif;
    }

    /* =========================================
       2. WIDGET VISIBILITY FIXES (CRITICAL)
       ========================================= */

    /* FIX: FILE UPLOADER (Remove Black Background) */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #f8fafc !important; /* Light Grey for dropzone */
        color: #000000 !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-testid="stFileUploader"] span {
        color: #475569 !important;
    }

    /* FIX: SELECTBOX, DATE INPUT, TEXT INPUT */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stDateInput"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* Input Text Color */
    input[type="text"], input[type="number"], .stDateInput input {
        color: #000000 !important;
    }

    /* Dropdown Options Container */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #ffffff !important;
    }
    li[role="option"] {
        color: #000000 !important;
    }

    /* =========================================
       3. SIDEBAR STYLING
       ========================================= */
    section[data-testid="stSidebar"] {
        background-color: var(--light-sky);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0369a1 !important;
    }

    /* =========================================
       4. TAB BACKGROUNDS (Specific Color Mapping)
       ========================================= */

    /* Tab Container Styling */
    div[data-testid="stTabPanel"] {
        padding: 25px;
        border-radius: 15px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.02);
    }

    /* Tab 1: Overview -> Frost Blue */
    div[data-testid="stTabPanel"]:nth-of-type(1) { background-color: var(--frost-blue); border: 1px solid #dbeafe; }

    /* Tab 2: Data Hub -> Vanilla */
    div[data-testid="stTabPanel"]:nth-of-type(2) { background-color: var(--vanilla); border: 1px solid #fef3c7; }

    /* Tab 3: Forecasting -> Soft Cyan */
    div[data-testid="stTabPanel"]:nth-of-type(3) { background-color: var(--soft-cyan); border: 1px solid #cffafe; }

    /* Tab 4: Health Metrics -> Sage Light */
    div[data-testid="stTabPanel"]:nth-of-type(4) { background-color: var(--sage-light); border: 1px solid #d1fae5; }

    /* Tab 5: Anomalies -> Light Red */
    div[data-testid="stTabPanel"]:nth-of-type(5) { background-color: var(--alert-red); border: 1px solid #ffe4e6; }

    /* Tab 6: Final Report -> Pastel Green */
    div[data-testid="stTabPanel"]:nth-of-type(6) { background-color: var(--pastel-green); border: 1px solid #dcfce7; }

    /* Active Tab Text Color */
    div[data-testid="stTabs"] button { color: #64748b !important; font-weight: 600; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #000000 !important;
        border-top-color: #000000 !important;
    }

    /* =========================================
       5. COMPONENT STYLING
       ========================================= */

    /* Metric Indicators (Gradient Cards from Image) */
    .metric-card {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); /* Adjusted to match image closer */
        background: linear-gradient(to right, #2b9db4, #44c69d); /* Exact Teal/Green Gradient */
        border-radius: 12px;
        padding: 25px;
        color: white !important;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }

    /* Force white text inside metric cards */
    .metric-card div, .metric-card span {
        color: #ffffff !important;
    }
    .metric-value { font-size: 32px; font-weight: 800; margin-bottom: 5px; }
    .metric-label { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.9; }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid #ffffff;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Health Card (Landing Page) */
    .health-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(to right, #2563eb, #0ea5e9);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ------------------ STATE MANAGEMENT ------------------
if "pipeline_stage" not in st.session_state: st.session_state["pipeline_stage"] = 0
if "data_summary" not in st.session_state: st.session_state["data_summary"] = None
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None
if "anomaly_results" not in st.session_state: st.session_state["anomaly_results"] = None
if "insight_results" not in st.session_state: st.session_state["insight_results"] = None
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050257.png", width=60)
    st.title("FitPulse Pro")
    st.markdown("---")

    st.markdown("### 📂 Data Import")
    # File Uploader - Now Styled White via CSS
    file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if file:
        st.caption(f"📄 {file.name}")
        if st.button("🚀 Launch Analysis", key="upload_btn"):
            with st.spinner("Ingesting & Preprocessing..."):
                try:
                    res = requests.post(f"{API_URL}/preprocess", files={"file": file.getvalue()})
                    if res.status_code == 200:
                        st.session_state["data_summary"] = res.json()
                        st.session_state["pipeline_stage"] = 1
                        st.success("Data Ready!")
                        st.rerun()
                    else:
                        st.error(f"Error: {res.json().get('error')}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

    if st.session_state["pipeline_stage"] > 0:
        if st.button("🔄 Reset System"):
            try: requests.post(f"{API_URL}/reset")
            except: pass
            st.session_state.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Status")
    stages = {1: "Preprocessing", 2: "Model building", 3: "Anomaly detection", 4: "Report generation"}
    curr = st.session_state.get("pipeline_stage", 0)
    for k, v in stages.items():
        icon = "🟢" if k <= curr else "⚪"
        st.write(f"{icon} **{v}**")

if st.session_state.get("pipeline_stage", 0) == 0:

    # 🎨 1. GLOBAL STYLING
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(to bottom right, #f8f9fa, #eef2f3); }
        .main-header { text-align: center; margin-bottom: 40px; }
        .main-header h1 { font-size: 3.5rem; color: #111827; font-weight: 800; letter-spacing: -1px; }
        .main-header p { font-size: 1.2rem; color: #6b7280; max-width: 700px; margin: 0 auto; }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            border: 1px solid #e5e7eb;
        }
        .info-box {
            background-color: #f3f4f6;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 0.9rem;
        }
        .highlight-red { color: #ef4444; font-weight: bold; }
        .highlight-green { color: #10b981; font-weight: bold; }
        .highlight-blue { color: #3b82f6; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # ------------------ HEADER ------------------
    st.markdown("""
    <div class="main-header">
        <h1>FitPulse Pro AI</h1>
        <p>Your intelligent biometric command center. Upload your data to detect anomalies, analyze sleep patterns, and optimize active living.</p>
    </div>
    """, unsafe_allow_html=True)

  # ------------------ UPDATED DEMO CHARTS (INTERACTIVE & STYLISH) ------------------
    def get_pro_heart_chart():
        # 1. Create Data: Smooth sine wave + sudden spike
        x = np.linspace(0, 10, 100)
        y = 70 + 5 * np.sin(x*2)  # Baseline
        y[45:55] += 50            # Spike (Tachycardia)

        # 2. Define Colors based on value
        colors = ['#ef4444' if val > 100 else '#6366f1' for val in y]

        fig = go.Figure()

        # 3. Add Line with Gradient Fill
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            fill='tozeroy', # Area chart style
            name='BPM',
            line=dict(color='#6366f1', width=3, shape='spline'), # Smooth curves
            fillcolor='rgba(99, 102, 241, 0.1)', # Light purple tint
            hovertemplate='<b>%{y:.0f} BPM</b><br>%{text}',
            text=['⚠️ Arrhythmia Detected' if val > 100 else '✅ Normal Rhythm' for val in y]
        ))

        # 4. Add Annotation for the Anomaly
        fig.add_annotation(
            x=5, y=125,
            text="Spike Detected (120 BPM)",
            showarrow=True,
            arrowhead=2,
            ax=0, ay=-40,
            bgcolor="#fee2e2", bordercolor="#ef4444", borderwidth=1,
            font=dict(color="#b91c1c", size=10)
        )

        fig.update_layout(
            template="plotly_white",
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, fixedrange=True),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', range=[50, 140]),
            showlegend=False,
            hovermode="x unified" # Clean hover line
        )
        return fig

    def get_pro_sleep_chart():
        fig = go.Figure()

        # Data: Comparison of "Your Sleep" vs "Ideal Sleep"
        stages = ["Deep (Restorative)", "Light/REM", "Awake"]
        user_vals = [15, 55, 30]  # User %
        ideal_vals = [25, 65, 10] # Ideal %
        colors = ['#4338ca', '#818cf8', '#e2e8f0'] # Dark Indigo, Light Indigo, Grey

        # Stacked Bars
        for i, stage in enumerate(stages):
            fig.add_trace(go.Bar(
                y=['<b>Your Sleep</b>', 'Target Guide'],
                x=[user_vals[i], ideal_vals[i]],
                name=stage,
                orientation='h',
                marker=dict(color=colors[i], line=dict(width=0)),
                hovertemplate=f"<b>{stage}</b>: %{{x}}%<extra></extra>"
            ))

        fig.update_layout(
            barmode='stack',
            template="plotly_white",
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, title="Sleep Composition (%)"),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color='#1f2937')),
            legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)),
            hovermode="y unified"
        )
        return fig

    def get_pro_steps_chart():
        # Gauge with semantic zones
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 4250,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Steps Today", 'font': {'size': 14, 'color': '#6b7280'}},
            delta = {'reference': 8000, 'increasing': {'color': "green"}, 'decreasing': {'color': "#ef4444"}, "position": "bottom", "relative": False},
            gauge = {
                'axis': {'range': [None, 10000], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': "#10b981", 'thickness': 0.15}, # Needle/Bar
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 5000], 'color': "rgba(239, 68, 68, 0.1)"},   # Red tint (Sedentary)
                    {'range': [5000, 8000], 'color': "rgba(234, 179, 8, 0.1)"}, # Yellow tint (Moderate)
                    {'range': [8000, 10000], 'color': "rgba(16, 185, 129, 0.1)"} # Green tint (Active)
                ],
                'threshold': {
                    'line': {'color': "#111827", 'width': 3}, # Target Line
                    'thickness': 0.8,
                    'value': 8000
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=25, r=25, t=0, b=0))
        return fig

    # ------------------ MAIN GRID (UPDATED) ------------------
    col1, col2, col3 = st.columns(3, gap="medium")

    # === COLUMN 1: HEART HEALTH ===
    with col1:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #ef4444; padding: 20px; height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h4 style="margin:0; color: #111827; font-weight:700;">❤️ Cardio Health</h4>
                <span style="background:#fef2f2; color:#ef4444; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:700; border:1px solid #fee2e2;">LIVE</span>
            </div>
            <p style="font-size: 0.85em; color: #020203; margin-bottom: 5px;">Rhythm Analysis & Anomaly Detection</p>
        """, unsafe_allow_html=True)

        st.plotly_chart(get_pro_heart_chart(), use_container_width=True, config={'displayModeBar': False})

        with st.expander("🔎 View Clinical Context", expanded=False):
            st.markdown("""
            <div style="font-size: 0.8em; color:#374151;">
                <p style="margin-bottom:5px;"><strong>Detected:</strong> <span style="color:#ef4444;">Arrhythmia Spike</span></p>
                <p style="margin-bottom:0;">Short-term tachycardia detected at rest. Correlation with caffeine intake or stress recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # === COLUMN 2: SLEEP HYGIENE ===
    with col2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #6366f1; padding: 20px; height: 100%;">
             <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h4 style="margin:0; color: #111827; font-weight:700;">😴 Sleep hours</h4>
                <span style="background:#e0e7ff; color:#4338ca; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:700; border:1px solid #c7d2fe;">AVG</span>
            </div>
            <p style="font-size: 0.85em; color: #020203; margin-bottom: 5px;">Deep Sleep vs Light/Awake Ratios</p>
        """, unsafe_allow_html=True)

        st.plotly_chart(get_pro_sleep_chart(), use_container_width=True, config={'displayModeBar': False})

        with st.expander("🔎 Optimization Tips", expanded=False):
            st.markdown("""
            <div style="font-size: 0.8em; color:#374151;">
                <p style="margin-bottom:5px;"><strong>Goal:</strong> <span style="color:#6366f1;">Increase Deep Sleep</span></p>
                <p style="margin-bottom:0;">Current Deep Sleep (15%) is below the 20% target. Reduce blue light exposure 60 mins before bed.</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # === COLUMN 3: ACTIVE LIVING ===
    with col3:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #10b981; padding: 20px; height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h4 style="margin:0; color: #111827; font-weight:700;">👣 Daily Mobility</h4>
                <span style="background:#5dcf92; color:#051b96; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:700; border:1px solid #a7f3d0;">TODAY</span>
            </div>
            <p style="font-size: 0.85em; color: #020203; margin-bottom: 5px;">Step Count & Sedentary Alerts</p>
        """, unsafe_allow_html=True)

        st.plotly_chart(get_pro_steps_chart(), use_container_width=True, config={'displayModeBar': False})

        with st.expander("🔎 Activity Impact", expanded=False):
            st.markdown("""
            <div style="font-size: 0.8em; color:#374151;">
                <p style="margin-bottom:5px;"><strong>Status:</strong> <span style="color:#d97706;">Sedentary Warning</span></p>
                <p style="margin-bottom:0;">You are 3,800 steps below the daily maintenance goal. A 20-min walk is recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ CTA ------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👉 **Ready to analyze your own data?** Upload your `health_data.csv` in the sidebar to unlock these insights for yourself.")

    st.stop()

# ------------------ MAIN DASHBOARD ------------------
# Tabs
tabs = st.tabs(["📊 Overview", "❤️ Data Hub", "🔮 Forecasting", "🏆 Health Metrics", "🚨 Anomalies", "🤖 AI & Report"])

# --- TAB 1: OVERVIEW (FROST BLUE) ---
with tabs[0]:
    st.markdown("### 📊 Executive Summary")

    if st.session_state["data_summary"]:
        # 1. Prepare Data
        df_sample = pd.DataFrame(st.session_state["data_summary"]["sample"])

        # Convert to numeric for calculations
        df_sample['heart_rate'] = pd.to_numeric(df_sample.get('heart_rate'), errors='coerce')
        df_sample['steps'] = pd.to_numeric(df_sample.get('steps'), errors='coerce')
        df_sample['sleep'] = pd.to_numeric(df_sample.get('sleep'), errors='coerce')
        df_sample['date'] = pd.to_datetime(df_sample.get('date'), errors='coerce').dt.date

        # Calculate Averages
        avg_hr = df_sample['heart_rate'].mean()
        avg_steps = df_sample['steps'].mean()
        avg_sleep = df_sample['sleep'].mean()

        # --- FEATURE 1: LIVE HEALTH STATUS PULSE ---
        score = 50
        if avg_sleep >= 7: score += 20
        if avg_steps >= 5000: score += 20
        if 60 <= avg_hr <= 100: score += 10

        # --- FEATURE 2: UPDATED METRIC CARDS WITH DELTAS ---
        # Logic: Compare vs standard health goals
        step_goal = 10000
        step_perf = (avg_steps / step_goal) * 100 if not pd.isna(avg_steps) else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{round(avg_hr,1) if not pd.isna(avg_hr) else '-'} <span style="font-size:16px">bpm</span></div>
                <div class="metric-label">Avg Heart Rate</div>
                <div style="color:#64748b; font-size:0.8em; margin-top:5px;">Normal Range: 60-100</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{int(avg_steps) if not pd.isna(avg_steps) else '-'} <span style="font-size:16px">steps</span></div>
                <div class="metric-label">Daily Average</div>
                <div style="color:{'#10b981' if step_perf > 70 else '#f59e0b'}; font-size:0.8em; margin-top:5px;">{'↑' if step_perf > 70 else '↓'} {round(step_perf, 1)}% of Goal</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{round(avg_sleep,1) if not pd.isna(avg_sleep) else '-'} <span style="font-size:16px">hrs</span></div>
                <div class="metric-label">Avg Sleep Duration</div>
                <div style="color:#64748b; font-size:0.8em; margin-top:5px;">Rec: 7-9 hours</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Advanced Visuals Row
        col_score, col_trend = st.columns([1, 2])

        with col_score:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {'text': "<b>Health Score</b>", 'font': {'size': 24, 'color': '#111827'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                    'bar': {'color': "#22d3ee"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                        {'range': [50, 80], 'color': 'rgba(234, 179, 8, 0.1)'},
                        {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                    ],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#111827"}, height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # --- FEATURE 3: INTERACTIVE TREND TOGGLE ---
        with col_trend:
            # Added a toggle to choose which metric to view in the trend
            chart_metric = st.radio("Trend View:", ["steps", "heart_rate", "sleep"], horizontal=True, label_visibility="collapsed")

            label_map = {"steps": "Steps Taken", "heart_rate": "Avg Heart Rate (bpm)", "sleep": "Sleep (hrs)"}
            color_map = {"steps": "#3b82f6", "heart_rate": "#ef4444", "sleep": "#8b5cf6"}

            if not df_sample.empty:
                df_sample = df_sample.sort_values('date')
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(
                    x=df_sample['date'],
                    y=df_sample[chart_metric],
                    marker_color=color_map[chart_metric],
                    name=label_map[chart_metric],
                    hovertemplate='%{y} ' + chart_metric + '<extra></extra>'
                ))

                fig_trend.update_layout(
                    title=f"📅 Recent {label_map[chart_metric]}",
                    template="plotly_white",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=260,
                    margin=dict(l=0,r=0,t=30,b=0),
                    font=dict(color="#111827"),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No data available for visualization.")
    else:
        st.info("Waiting for data upload...")

# --- TAB 2: DATA HUB (VANILLA) ---
with tabs[1]:
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #22d3ee;">
        <h3 style="margin:0; color: #111827;">🧪 Data Collection & Preprocessing Hub</h3>
        <p style="margin:0; color: #4b5563;">Enterprise-grade validation, inspection, and export pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state["data_summary"]

    if data:
        c_left, c_right = st.columns([2, 1])

        with c_left:
            st.markdown("#### 📄 Live Dataset Preview")
            if data.get("sample") and len(data["sample"]) > 0:
                try:
                    df_prev = pd.DataFrame(data["sample"])
                    df_display = df_prev.fillna("N/A").astype(str)
                    st.dataframe(df_display, use_container_width=True, height=400, hide_index=True)
                except Exception as e:
                    st.error(f"Rendering Error: {e}")
            else:
                st.warning("⚠️ Backend returned 0 records.")

        with c_right:
            st.markdown("#### 🧠 Data Schema")
            cols = data.get("columns") or (list(data["sample"][0].keys()) if data.get("sample") else [])
            st.json(cols, expanded=False)

            st.markdown("#### ⬇️ Actions")
            if st.button("📥 Download Cleaned CSV"):
                try:
                    res = requests.get(f"{API_URL}/download_clean_csv")
                    if res.status_code == 200:
                        payload = res.json()
                        csv_bytes = base64.b64decode(payload["csv_file"])
                        st.download_button("💾 Save CSV File", csv_bytes, "fitpulse_data.csv", "text/csv", use_container_width=True)
                    else: st.error("Download failed.")
                except Exception as e: st.error(f"Connection error: {e}")
    else:
        st.info("Please upload a file in the sidebar to view the Data Hub.")

# --- TAB 3: FORECASTING (SOFT CYAN) ---
with tabs[2]:
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #0891b2;">
        <h3 style="margin:0; color:#111827;">🔮 Personal Health Predictor</h3>
        <p style="margin:0; color:#4b5563;">Select a user to generate a personalized health forecast for the upcoming days.</p>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.get("data_summary")

    if data:
        # Get Users
        users = [str(u) for u in data.get("users", [])]
        if not users and data.get("sample"):
            users = sorted([str(u) for u in pd.DataFrame(data["sample"])["user_id"].unique().tolist()])

        c_user, c_date, c_action = st.columns([1, 1, 1])

        with c_user:
            selected_user = st.selectbox("Select User ID", ["All"] + users)

        with c_date:
            # Default to 7 days from today
            default_date = datetime.date.today() + datetime.timedelta(days=7)
            target_date = st.date_input("Forecast Until", value=default_date)

        with c_action:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶ Run Forecast", key="run_fc"):

                # --- FIX: Send 'target_date' as a STRING ---
                # This matches your new Backend 'ForecastRequest' model
                req_body = {
                    "user_id": str(selected_user),
                    "target_date": str(target_date)
                }

                with st.spinner(f"Generating AI Forecast for User {selected_user}..."):
                    try:
                        res = requests.post(f"{API_URL}/module2", json=req_body)

                        if res.status_code == 200:
                            st.session_state["analysis_results"] = res.json()
                            st.rerun()
                        else:
                            # FIX: Properly read the error if the backend rejects the request
                            err_data = res.json()
                            # FastAPI validation errors use 'detail', custom errors use 'error'
                            err_msg = err_data.get('error') or err_data.get('detail')
                            st.error(f"Backend Error: {err_msg}")

                    except Exception as e:
                        st.error(f"Connection Error: {e}")

        # --- RESULTS DISPLAY ---
        if st.session_state.get("analysis_results"):
            res = st.session_state["analysis_results"]

            # 1. Visualization
            if res.get("forecast_chart"):
                st.markdown("#### 📈 AI Trend Projection")
                fig_fc = pio.from_json(res["forecast_chart"])
                fig_fc.update_layout(
                    template="plotly_white",
                    font=dict(color="#111827"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_fc, use_container_width=True)

            # 2. Forecasting Table
            if res.get("forecast_data"):
                st.markdown("#### 📋 Predicted Metrics Table")
                forecast_df = pd.DataFrame(res["forecast_data"])

                # Rename for UI clarity
                cols_map = {"ds": "Date", "yhat": "Predicted HR", "yhat_lower": "Min Expected", "yhat_upper": "Max Expected"}
                forecast_df = forecast_df.rename(columns=cols_map)

                # Reorder columns nicely
                desired_order = ["Date", "Predicted HR", "Min Expected", "Max Expected"]
                # Only select columns that actually exist to prevent errors
                available_cols = [c for c in desired_order if c in forecast_df.columns]
                forecast_df = forecast_df[available_cols]

                st.dataframe(
                    forecast_df.style.format({"Predicted HR": "{:.1f}", "Min Expected": "{:.1f}", "Max Expected": "{:.1f}"}),
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info("Please upload a file in the 'Data Upload' tab to begin.")

# --- TAB 4: HEALTH METRICS (SAGE LIGHT) ---
with tabs[3]:
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #10b981;">
        <h3 style="margin:0; color:#111827;">🏆 Wellness Score & Metrics</h3>
        <p style="margin:0; color:#4b5563;">Deep dive into your aggregated health scores and averages.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Calculate Metrics", key="calc_metrics_btn"):
         with st.spinner("Analyzing User Data..."):
             try:
                 res = requests.post(f"{API_URL}/module4")
                 if res.status_code == 200:
                     st.session_state["insight_results"] = res.json()
                     st.session_state["pipeline_stage"] = 4
                     st.rerun()
                 else:
                     st.error("Failed to generate metrics.")
             except Exception as e:
                 st.error(f"Error: {e}")

    # Display Content if available
    if st.session_state.get("insight_results"):
        res = st.session_state["insight_results"]
        avgs = res.get("averages", {})

        # 1. Cards Row (Styled to match image)
        m1, m2, m3 = st.columns(3)

        with m1:
             val = f"{avgs.get('avg_heart_rate', '-')}"
             st.markdown(f"""<div class="metric-card"><div class="metric-value">{val} <span style="font-size:16px">bpm</span></div><div class="metric-label">Avg Heart Rate</div></div>""", unsafe_allow_html=True)

        with m2:
             val = f"{avgs.get('avg_steps', '-')}"
             st.markdown(f"""<div class="metric-card"><div class="metric-value">{val} <span style="font-size:16px">steps</span></div><div class="metric-label">Daily Average</div></div>""", unsafe_allow_html=True)

        with m3:
             val = f"{avgs.get('avg_sleep', '-')}"
             st.markdown(f"""<div class="metric-card"><div class="metric-value">{val} <span style="font-size:16px">hrs</span></div><div class="metric-label">Avg Sleep Duration</div></div>""", unsafe_allow_html=True)

        # 2. Gauge Chart Row (Centered below cards)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🩺 Overall Health Score")
        if res.get("gauge_chart"):
            fig_g = pio.from_json(res["gauge_chart"])
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#111827"})
            st.plotly_chart(fig_g, use_container_width=True)

    else:
        st.info("Click 'Calculate Metrics' to view your health score.")


# --- TAB 5: ANOMALIES (LIGHT RED/ALERT THEME) ---
with tabs[4]:
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #ef4444;">
        <h3 style="margin:0; color:#111827;">🚨 Anomaly Detection Engine</h3>
        <p style="margin:0; color:#4b5563;">Scans for Tachycardia, Sleep Deprivation, and unusual multivariate patterns (DBSCAN).</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧠 Start Deep Scan", key="ano_btn"):
        with st.spinner("Initializing Feature Engineering & Rules Engine..."):
            try:
                res = requests.post(f"{API_URL}/module3")
                if res.status_code == 400 and "Module 2" in res.text:
                    st.toast("⚙️ Generating features...", icon="⚠️")
                    requests.post(f"{API_URL}/module2", json={"user_id": "All", "days": 10})
                    res = requests.post(f"{API_URL}/module3")

                if res.status_code == 200:
                    st.session_state["anomaly_results"] = res.json()
                    st.session_state["pipeline_stage"] = max(st.session_state["pipeline_stage"], 3)
                    st.rerun()
                else:
                    st.error(f"Scan Failed: {res.json().get('error')}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

    if st.session_state.get("anomaly_results"):
        res = st.session_state["anomaly_results"]
        charts = res.get("charts", {})
        if charts:
            st.markdown("#### 📉 Anomaly Visualizations")

            # 1. Heart Rate (Top)
            if charts.get("hr_chart"):
                fig_hr = pio.from_json(charts["hr_chart"])
                fig_hr.update_layout(template="plotly_white", font=dict(color="#111827"), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hr, use_container_width=True)

            # 2. Sleep hours (Middle)
            if charts.get("sleep_chart"):
                fig_sl = pio.from_json(charts["sleep_chart"])
                fig_sl.update_layout(template="plotly_white", font=dict(color="#111827"), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sl, use_container_width=True)

            # 3. Daily Steps (Bottom)
            if charts.get("steps_chart"):
                fig_st = pio.from_json(charts["steps_chart"])
                fig_st.update_layout(template="plotly_white", font=dict(color="#111827"), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_st, use_container_width=True)

        st.markdown("#### ⚠️ Event Log")
        alerts = res.get("alerts", [])
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.success("✅ System Normal: No anomalies detected in the dataset.")
    else:
        st.info("Click 'Start Deep Scan' to analyze the data.")

# --- TAB 6: AI & REPORT (PASTEL GREEN) ---
with tabs[5]:
    st.markdown("### 📋 Clinical Intelligence Dashboard")

    # Create THREE sub-tabs for organized workflow
    sub_tabs = st.tabs(["📊 Insights & Report", "🩺 Symptom Checker", "👨‍⚕️ AI Doctor"])

    # ---------------------------------------------------------
    # SUB-TAB 1: ANALYSIS & REPORT
    # ---------------------------------------------------------
    with sub_tabs[0]:
        col1, col2 = st.columns([1, 3])

        # Left Sidebar: Actions
        with col1:
            st.markdown("#### Actions")
            if st.button("✨ Refresh Insights", key="gen_rep_btn", use_container_width=True):
                 with st.spinner("Analyzing Clinical Data..."):
                     try:
                         res = requests.post(f"{API_URL}/module4")
                         if res.status_code == 200:
                             st.session_state["insight_results"] = res.json()
                             st.rerun()
                         else:
                             st.error("Please run Modules 1-3 first.")
                     except Exception as e:
                         st.error(f"Connection Error: {e}")

            if st.session_state.get("insight_results"):
                st.markdown("---")
                st.caption("Export Data")
                if st.button("📥 Download Report", key="dl_txt_btn", use_container_width=True):
                    try:
                        dl_res = requests.get(f"{API_URL}/download_insights")
                        if dl_res.status_code == 200:
                            b64 = dl_res.json()["file_content"]
                            st.download_button(
                                label="💾 Save as Text",
                                data=base64.b64decode(b64),
                                file_name="Health_Report.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                    except: st.warning("Report not ready.")

        # Right Panel: Insights Feed
        with col2:
            st.markdown("#### 📝 Clinical Insights Feed")
            if st.session_state.get("insight_results"):
                res = st.session_state["insight_results"]
                insights_list = res.get("insights", [])

                if insights_list:
                    for item in insights_list:
                        sev = item.get('severity', 'Low')
                        colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22d3ee"}
                        border_color = colors.get(sev, "#22d3ee")

                        st.markdown(f"""
                        <div style="border-left: 5px solid {border_color}; padding: 15px; margin-bottom:12px; background: white; border-radius: 4px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <strong style="color:#1f2937;">User: {item.get('user_id')}</strong>
                                <span style="background:{border_color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.75em; font-weight:600;">{sev.upper()}</span>
                            </div>
                            <div style="color:#4b5563; font-size:0.95em; line-height:1.4;">
                                {item.get('insight').replace('**', '<b>').replace('**', '</b>').replace('\n', '<br>')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("✅ No critical anomalies detected.")
            else:
                st.info("Click 'Refresh Insights' to generate the analysis.")

    # ---------------------------------------------------------
    # SUB-TAB 2: SYMPTOM CHECKER
    # ---------------------------------------------------------
    with sub_tabs[1]:
        st.markdown("#### 🩺 AI Symptom Checker")
        c1, c2 = st.columns([1, 1])
        with c1:
            common_symptoms = ["Fatigue", "Dizziness", "Chest Pain", "Shortness of Breath", "Insomnia", "Headache", "Palpitations", "Nausea", "Fever", "Anxiety"]
            selected_symptoms = st.multiselect("Select Symptoms:", common_symptoms)
        with c2:
            notes = st.text_area("Additional Notes:", placeholder="Started 2 days ago...")

        if st.button("🔍 Analyze Symptoms", use_container_width=True):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                with st.spinner("Consulting AI..."):
                    try:
                        payload = {"symptoms": selected_symptoms, "additional_notes": notes}
                        res = requests.post(f"{API_URL}/check_symptoms", json=payload)
                        if res.status_code == 200:
                            st.session_state["symptom_report"] = res.json().get("report", "No report generated.")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")

        if "symptom_report" in st.session_state:
            st.markdown("---")
            st.markdown("### 🏥 Generated Health Report")
            st.markdown(f"""<div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb;">{st.session_state['symptom_report']}</div>""", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SUB-TAB 3: AI DOCTOR (NEW CHATBOT LOCATION)
    # ---------------------------------------------------------
    with sub_tabs[2]:
        st.markdown("#### 👨‍⚕️ AI Doctor")
        st.caption("Ask questions about your data, trends, or get general health advice.")

        # Chat Container
        chat_box = st.container(height=450)

        # Display History
        with chat_box:
            if not st.session_state["chat_history"]:
                st.markdown("""
                <div style="text-align:center; color:#9ca3af; padding-top:50px;">
                    <b>Welcome! I am your AI Doctor.</b><br>
                    I have access to your health logs and can help interpret them.<br>
                    <i>Try: "Is my average heart rate normal?"</i>
                    <i>Try: "How can I improve my health?"</i>

                </div>
                """, unsafe_allow_html=True)

            for role, text in st.session_state["chat_history"]:
                avatar = "🤖" if role == "assistant" else "👤"
                bg = "#f3f4f6" if role == "assistant" else "#e0f2fe"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(f"""<div style="background:{bg}; padding:10px; border-radius:8px; color:#1f2937;">{text}</div>""", unsafe_allow_html=True)

        # Input Area (Within this tab)
        if prompt := st.chat_input("Message the AI Doctor..."):
            st.session_state["chat_history"].append(("user", prompt))

            with chat_box:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"<div style='background:#e0f2fe; padding:10px; border-radius:8px; color:#1f2937;'>{prompt}</div>", unsafe_allow_html=True)

                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Consulting medical records...")
                    try:
                        api_res = requests.post(f"{API_URL}/ask_ai", json={"question": prompt})
                        ans = api_res.json().get("answer", "I couldn't process that.") if api_res.status_code == 200 else "Server Error."
                    except:
                        ans = "Connection Error."

                    message_placeholder.markdown(f"<div style='background:#f3f4f6; padding:10px; border-radius:8px; color:#1f2937;'>{ans}</div>", unsafe_allow_html=True)
                    st.session_state["chat_history"].append(("assistant", ans))
                    st.rerun() # Ensure UI refreshes to show the message