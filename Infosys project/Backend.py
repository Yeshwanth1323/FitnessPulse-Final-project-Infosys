
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import base64
from typing import Optional, List
import nest_asyncio
import datetime
nest_asyncio.apply() # Apply nest_asyncio at the earliest point

# --- IMPORTS FOR CHARTING & ML ---
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from prophet import Prophet

# --- TSFRESH IMPORTS ---
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

# --- IMPORT GOOGLE GEMINI ---
from google import genai
from google.genai import types

# ==========================================
# SETUP & CONFIGURATION
# ==========================================
app = FastAPI(title="FitPulse Pro – Gemini Powered")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# *** CONFIGURE GEMINI HERE ***
# Paste your key from aistudio.google.com below
GEMINI_API_KEY = "gemini api key"

client = None
try:
    if GEMINI_API_KEY and "YOUR_GEMINI" not in GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini AI Client successfully connected.")
    else:
        print("⚠️ Gemini Key missing. Chat features will be disabled.")
except Exception as e:
    print(f"❌ Error connecting to Gemini: {e}")

# Global In-Memory Store
DATA_STORE = {
    "clean": None,
    "features": None,
    "alerts": None,
    "insights_text": "No analysis run yet."
}

# Request Schemas
class ChatRequest(BaseModel):
    question: str

# UPDATED: Request accepts target_date string instead of fixed days
class ForecastRequest(BaseModel):
    user_id: Optional[str] = "All"
    target_date: str

class SymptomRequest(BaseModel):
    symptoms: List[str]
    additional_notes: Optional[str] = ""

def robust_standardize(df):
    mapping = {
        "user_id": ["Patient_ID", "User_ID", "user_id", "ID", "id"],
        "date": ["date", "Date", "timestamp", "DateTime", "ActivityDate", "time"],
        "steps": ["TotalSteps", "daily_steps", "Steps_Taken", "step_count", "steps"],
        "heart_rate": ["avg_heart_rate", "heart_rate", "Heart_Rate (bpm)", "value", "bpm"],
        "sleep": ["sleep_hours", "daily_sleep_hours", "Hours_Slept", "sleep_duration", "total_sleep_minutes", "minutesAsleep"],
        "bmi": ["BMI", "bmi", "BodyMassIndex"]
    }
    new_cols = {}
    for target, variations in mapping.items():
        for var in variations:
            if var in df.columns:
                new_cols[var] = target
                break
    df = df.rename(columns=new_cols)
    if "date" not in df.columns: return None, "Missing mandatory 'date' column."
    return df, None

# ==========================================
# MODULE 1: PREPROCESSING
# ==========================================
@app.post("/preprocess")
async def preprocess_data(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # 1. Standardize
        df, error = robust_standardize(df)
        if error: return JSONResponse(status_code=400, content={"error": error})

        # 2. Clean Dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.date

        # 3. Defaults & TYPE CASTING
        if "user_id" not in df.columns: df["user_id"] = "User_1"
        df["user_id"] = df["user_id"].astype(str)

        # 4. Numeric Conversion
        for col in ["steps", "heart_rate", "sleep"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

        # 5. Missing Values
        if "steps" in df.columns: df["steps"] = df["steps"].fillna(0)
        if "heart_rate" in df.columns: df["heart_rate"] = df["heart_rate"].fillna(df["heart_rate"].median())
        if "sleep" in df.columns:
            if df["sleep"].mean() > 24: df["sleep"] = df["sleep"] / 60
            df["sleep"] = df["sleep"].fillna(df["sleep"].median())

        # 6. Aggregate
        agg_logic = {}
        if "steps" in df.columns: agg_logic["steps"] = "max"
        if "sleep" in df.columns: agg_logic["sleep"] = "mean"
        if "heart_rate" in df.columns: agg_logic["heart_rate"] = "mean"

        if agg_logic:
            df = df.groupby(["user_id", "date"], as_index=False).agg(agg_logic)

        DATA_STORE["clean"] = df
        unique_users = sorted(df["user_id"].unique().tolist())

        return {
            "status": "success",
            "rows": len(df),
            "columns": df.columns.tolist(),
            "users": unique_users,
            "sample": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
            "max_date": str(df["date"].max())
        }
    except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

# ==========================================
# MODULE 2: TSFRESH & FORECASTING (FIXED)
# ==========================================
@app.post("/module2")
def module2(req: ForecastRequest):
    df = DATA_STORE["clean"]
    if df is None: return JSONResponse(status_code=400, content={"error": "Run Module 1 first"})

    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["user_id"] = df["user_id"].astype(str)
        target_user = str(req.user_id)

        # 1. FILTER FOR SPECIFIC USER
        if target_user and target_user != "All":
            df = df[df["user_id"] == target_user]

        if df.empty: return JSONResponse(status_code=400, content={"error": f"No data found for user: {target_user}"})

        df = df.sort_values(["user_id", "date"])

        # --- DYNAMIC DATE CALCULATION LOGIC ---
        # Find the last date in the existing dataset
        last_data_date = df["date"].max()

        # Parse the user's requested target date
        try:
            user_target_date = pd.to_datetime(req.target_date)
        except:
            # Fallback if parsing fails
            user_target_date = last_data_date + datetime.timedelta(days=7)

        # Calculate difference: Target Date - Last Known Data Date
        delta_days = (user_target_date - last_data_date).days

        # Ensure we predict at least 7 days if the user selects a past date or same date
        days_to_predict = delta_days if delta_days > 0 else 7

        print(f"DEBUG: Forecast Duration Calculated: {days_to_predict} days (Until {user_target_date.date()})")

        # ---------------------------------------------------------
        # 2. TSFRESH FEATURE EXTRACTION
        # ---------------------------------------------------------
        target_cols = [c for c in ["heart_rate", "steps", "sleep"] if c in df.columns]

        # Only run TSFresh if we have sufficient history to avoid errors
        if len(target_cols) > 0 and len(df) > 5:
            try:
                df_rolled = roll_time_series(
                    df,
                    column_id="user_id",
                    column_sort="date",
                    max_timeshift=5,
                    min_timeshift=2
                )

                X = extract_features(
                    df_rolled,
                    column_id="id",
                    column_sort="date",
                    column_value=target_cols[0],
                    default_fc_parameters=MinimalFCParameters(),
                    n_jobs=0
                )

                impute(X)
                X = X.reset_index()
                if "level_1" in X.columns:
                    X = X.rename(columns={"level_1": "date"})

                X["date"] = pd.to_datetime(X["date"])
                df = pd.merge(df, X, on="date", how="left")
                df = df.fillna(0)

            except Exception as e:
                print(f"TSFresh skipped due to: {e}")

        # ---------------------------------------------------------
        # 3. PROPHET FORECASTING (Dynamic Duration)
        # ---------------------------------------------------------
        forecast_table = []
        forecast_chart = None

        if "heart_rate" in df.columns and len(df) > 5:
            try:
                p_df = df.groupby("date")["heart_rate"].mean().reset_index().rename(columns={"date": "ds", "heart_rate": "y"})
                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
                m.fit(p_df)

                # USE CALCULATED 'days_to_predict'
                future = m.make_future_dataframe(periods=days_to_predict)
                forecast = m.predict(future)

                # EXTRACT ONLY THE FUTURE PREDICTIONS (Tail)
                forecast_res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_predict)
                forecast_res['ds'] = forecast_res['ds'].dt.strftime('%Y-%m-%d')
                forecast_table = forecast_res.to_dict("records")

                # Chart
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig_fc.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(34, 211, 238, 0.2)', name='Confidence'))
                fig_fc.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat'], mode='lines+markers', name='Prediction', line=dict(color='#22d3ee', width=3)))

                # Add a marker for the current status (last real data point) to connect the lines
                last_real_val = df[df["date"] == last_data_date]["heart_rate"].mean()
                fig_fc.add_trace(go.Scatter(
                    x=[last_data_date.strftime('%Y-%m-%d')],
                    y=[last_real_val],
                    mode='markers', name='Last Observed',
                    marker=dict(color='white', size=6, line=dict(width=2, color='#22d3ee'))
                ))

                fig_fc.update_layout(title=f"Forecast until {user_target_date.date()} | User {target_user}", template="plotly_dark", height=400)
                forecast_chart = pio.to_json(fig_fc)
            except Exception as e:
                print(f"Prophet Error: {e}")

        # ---------------------------------------------------------
        # 4. DBSCAN CLUSTERING
        # ---------------------------------------------------------
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_drop = ["pca_1", "pca_2", "is_ml_anomaly", "user_id"]
        cluster_cols = [c for c in numeric_cols if c not in cols_to_drop]

        clustering_chart = None

        if len(cluster_cols) >= 1:
            try:
                X_cluster = df[cluster_cols].fillna(0)
                X_scaled = StandardScaler().fit_transform(X_cluster)

                db = DBSCAN(eps=1.5, min_samples=3)
                df["is_ml_anomaly"] = db.fit_predict(X_scaled) == -1

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                df_pca = df.copy()
                df_pca["pca_1"] = X_pca[:, 0]
                df_pca["pca_2"] = X_pca[:, 1]
                df_pca["Status"] = np.where(df_pca["is_ml_anomaly"], "Anomaly", "Normal")

                fig_cl = px.scatter(
                    df_pca, x="pca_1", y="pca_2", color="Status",
                    color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#ef4444"},
                    symbol="Status", title=f"Cluster Analysis | User {target_user}"
                )
                fig_cl.update_layout(template="plotly_dark", height=400)
                clustering_chart = pio.to_json(fig_cl)
            except: pass
        else:
            df["is_ml_anomaly"] = False

        DATA_STORE["features"] = df

        return {
            "status": "success",
            "anomalies_detected": int(df["is_ml_anomaly"].sum()) if "is_ml_anomaly" in df.columns else 0,
            "forecast_data": forecast_table,
            "forecast_chart": forecast_chart,
            "clustering_chart": clustering_chart
        }
    except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

# ==========================================
# MODULE 3: VISUALIZATION & ANOMALY DETECTION
# ==========================================
@app.post("/module3")
def module3():
    df = DATA_STORE["features"]
    if df is None: return JSONResponse(status_code=400, content={"error": "Run Module 2 first"})

    try:
        df = df.copy()
        rows = []
        for _, r in df.iterrows():
            reason = None
            severity = "Low"
            if "heart_rate" in r:
                if r["heart_rate"] > 120: reason, severity = "Severe Tachycardia (>120)", "High"
                elif r["heart_rate"] > 90: reason, severity = "Elevated Resting HR (>90)", "Medium"
                elif r["heart_rate"] < 40: reason, severity = "Bradycardia (<40)", "High"
            if "sleep" in r:
                if r["sleep"] < 4: reason, severity = "Severe Sleep Deprivation (<4h)", "High"
                elif r["sleep"] < 6: reason, severity = "Insufficient Sleep (<6h)", "Low"
                elif r["sleep"] > 12: reason, severity = "Hypersomnia (>12h)", "Medium"
            if "steps" in r and r["steps"] < 3000: reason, severity = "Sedentary Behavior (<3k steps)", "Medium"
            if r.get("is_ml_anomaly", False): reason, severity = "Unusual Pattern (DBSCAN)", "Medium"
            if reason: rows.append((r["user_id"], r["date"], reason, severity))

        alert_logs = pd.DataFrame(rows, columns=["user_id", "date", "reason", "severity"])
        alerts_summary = alert_logs.groupby(["user_id", "reason", "severity"]).size().reset_index(name='count') if not alert_logs.empty else pd.DataFrame(columns=["user_id", "reason", "severity", "count"])
        DATA_STORE["alerts"] = alerts_summary

        charts = {}
        # 1. Heart Rate
        if "heart_rate" in df.columns:
            df['rolling_mean'] = df['heart_rate'].rolling(window=7, min_periods=1).mean()
            df['rolling_std'] = df['heart_rate'].rolling(window=7, min_periods=1).std().fillna(0)
            df['upper_band'] = df['rolling_mean'] + (2 * df['rolling_std'])
            df['lower_band'] = df['rolling_mean'] - (2 * df['rolling_std'])

            fig_hr = go.Figure()
            fig_hr.add_trace(go.Scatter(x=df['date'], y=df['upper_band'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_hr.add_trace(go.Scatter(x=df['date'], y=df['lower_band'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='Expected Range'))
            fig_hr.add_trace(go.Scatter(x=df['date'], y=df['heart_rate'], mode='lines', name='Heart Rate', line=dict(color='royalblue', width=2)))
            anoms = df[df["is_ml_anomaly"] == True]
            if not anoms.empty: fig_hr.add_trace(go.Scatter(x=anoms['date'], y=anoms['heart_rate'], mode='markers', marker=dict(color='red', size=10, symbol='x')))

            fig_hr.update_layout(title="Heart Rate Anomaly Detection", template="plotly_dark", height=400)
            charts["hr_chart"] = pio.to_json(fig_hr)

        # 2. Sleep
        if "sleep" in df.columns:
            def get_sleep_color(val):
                return "Critical (<4h)" if val < 4 else "Low (<6h)" if val < 6 else "Normal"

            df['sleep_status'] = df['sleep'].apply(get_sleep_color)

            neon_colors = {
                "Normal": "#00FF7F",
                "Low (<6h)": "#FFD700",
                "Critical (<4h)": "#FF3333"
            }

            fig_sleep = px.bar(
                df, x='date', y='sleep', color='sleep_status',
                color_discrete_map=neon_colors, title="Sleep hours"
            )
            fig_sleep.update_traces(marker_line_width=0)
            fig_sleep.add_hline(y=8, line_dash="dot", line_color="#FFFFFF", annotation_text="Ideal (8h)")
            fig_sleep.update_layout(template="plotly_dark", height=400)
            charts["sleep_chart"] = pio.to_json(fig_sleep)

        # 3. Steps
        if "steps" in df.columns:
            fig_s = go.Figure(go.Scatter(x=df['date'], y=df['steps'], fill='tozeroy', line=dict(color='#7e34d3')))
            fig_s.update_layout(title="Daily Steps", template="plotly_dark", height=400)
            charts["steps_chart"] = pio.to_json(fig_s)

            df['sleep_status'] = df['sleep'].apply(get_sleep_color)

            neon_colors = {
                "Normal": "#7e34d3",
                "Low (<3000)": "#FFD700",
            }

        return {"status": "success", "alerts": alerts_summary.to_dict("records"), "charts": charts}
    except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

# ==========================================
# MODULE 4: INSIGHTS & GAUGE
# ==========================================
@app.post("/module4")
def module4():
    df = DATA_STORE["features"]
    alerts_df = DATA_STORE["alerts"]
    if df is None: return JSONResponse(status_code=400, content={"error": "Run Module 2 first"})

    # 1. Averages & Wellness Score
    avg_stats = {}
    total_score_components = []

    if "steps" in df.columns:
        avg = df["steps"].mean()
        avg_stats["avg_steps"] = int(avg)
        total_score_components.append(min((avg / 10000) * 100, 100))

    if "sleep" in df.columns:
        avg = df["sleep"].mean()
        avg_stats["avg_sleep"] = round(avg, 1)
        total_score_components.append(min((avg / 8) * 100, 100))

    if "heart_rate" in df.columns:
        avg = df["heart_rate"].mean()
        avg_stats["avg_heart_rate"] = int(avg)
        dist = abs(avg - 65)
        total_score_components.append(max(100 - (dist * 2), 0))

    wellness_score = int(np.mean(total_score_components)) if total_score_components else 50

    # 2. Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = wellness_score,
        title = {'text': "Wellness Score"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#22d3ee"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [{'range': [0, 50], 'color': '#333'}, {'range': [50, 80], 'color': '#444'}]
        }
    ))
    fig_gauge.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=50, b=20))
    gauge_chart = pio.to_json(fig_gauge)

    # 3. Context Generation for AI (Averages + Raw Data)
    summary_lines = ["FitPulse Analysis Log", "========================"]
    summary_lines.append(f"Wellness Score: {wellness_score}/100")
    for k, v in avg_stats.items(): summary_lines.append(f"{k}: {v}")

    # Add Alerts
    rec_map = {
        "Severe Tachycardia (>120)": "Medical Alert: HR critically high.",
        "Elevated Resting HR (>90)": "Stress Management: Guided breathing.",
        "Bradycardia (<40)": "Medical Consultation: Rule out heart block.",
        "Severe Sleep Deprivation (<4h)": "Recovery Mode: Prioritize sleep.",
        "Insufficient Sleep (<6h)": "Sleep Hygiene: Advance bedtime 30 mins.",
        "Sedentary Behavior (<3k steps)": "Move More: Take 5 min walks hourly."
    }

    insights = []
    if alerts_df is not None and not alerts_df.empty:
        for _, row in alerts_df.iterrows():
            rec = rec_map.get(row['reason'], "Monitor closely.")
            insights.append({"user_id": str(row['user_id']), "severity": row['severity'], "insight": f"**{row['reason']}**\n{rec}"})
            summary_lines.append(f"- {row['reason']}: {rec}")
    else:
        insights.append({"user_id": "System", "severity": "Low", "insight": "No anomalies detected."})

    # ADD RAW DATA SAMPLE FOR AI
    raw_sample = df.tail(5).to_string(index=False)
    final_context = f"SUMMARY:\n" + "\n".join(summary_lines) + f"\n\nRECENT LOGS:\n{raw_sample}"
    DATA_STORE["insights_text"] = final_context

    return {"insights": insights, "averages": avg_stats, "gauge_chart": gauge_chart}

# ==========================================

# MODULE 5: GEMINI AI ASSISTANT & SYMPTOM CHECKER

# ==========================================

@app.post("/ask_ai")

async def ask_ai(request: ChatRequest):

    if not client:

        return {"answer": "AI is not active. Please add a valid Gemini API Key in backend.py."}



    context = DATA_STORE.get("insights_text", "No health data analysis available yet.")



    system_instruction = (

        "You are FitPulse AI, a helpful health assistant. "

        "Use the provided USER DATA to answer questions about the specific user's health metrics. "

        "Keep answers concise, friendly, and data-driven."

    )



    prompt = f"USER DATA:\n{context}\n\nUSER QUESTION:\n{request.question}"



    try:

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt,

            config=types.GenerateContentConfig(

                system_instruction=system_instruction,

                temperature=0.7

            )

        )

        return {"answer": response.text}

    except Exception as e:

        return {"error": f"Gemini Error: {str(e)}"}



@app.post("/check_symptoms")

async def check_symptoms(request: SymptomRequest):

    if not client:

        return {"report": "AI is not active. Please check API Key."}



    symptoms_list = ", ".join(request.symptoms)

    notes = request.additional_notes



    # Check if we have user vitals to add context

    user_context = DATA_STORE.get("insights_text", "No recent vitals available.")



    prompt = f"""

    You are an AI Medical Assistant acting as a preliminary symptom checker.



    PATIENT SYMPTOMS: {symptoms_list}

    ADDITIONAL NOTES: {notes}



    PATIENT VITALS (CONTEXT):

    {user_context}



    TASK:

    Generate a professional, structured health report.

    1. **Possible Causes**: List 3 potential causes based on symptoms.

    2. **Recommendation**: Suggest immediate actions (e.g., rest, hydration, see a doctor).

    3. **Warning Signs**: When to seek emergency care immediately.

    4. **Disclaimer**: End with a standard medical disclaimer (Not a doctor).



    Format nicely with Markdown.

    """



    try:

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt,

            config=types.GenerateContentConfig(temperature=0.4)

        )

        return {"report": response.text}

    except Exception as e:

        return {"error": f"Gemini Error: {str(e)}"}

# ==========================================
# UTILS
# ==========================================
@app.post("/reset")
def reset():
    global DATA_STORE
    DATA_STORE = {"clean": None, "features": None, "alerts": None, "insights_text": ""}
    return {"status": "success"}

@app.get("/download_clean_csv")
def download_clean_csv():
    df = DATA_STORE.get("clean")
    if df is None: return JSONResponse(status_code=400, content={"error": "No data"})
    return {"filename": "data.csv", "csv_file": base64.b64encode(df.to_csv(index=False).encode()).decode()}

@app.get("/download_insights")
def download_insights():
    text = DATA_STORE.get("insights_text", "No report available.")
    return {"filename": "report.txt", "file_content": base64.b64encode(text.encode()).decode()}

# Remove or simplify this block as the server is started in a separate thread.
# if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)

# For Colab, the Uvicorn server is often started in a separate thread/process.
# The `if __name__ == "__main__":` block will not be executed when the module is imported.
# If you were to run this backend.py file directly, this block would execute.
# Since we're running it via `uvicorn.run("backend:app", ...)` in another cell, it's not needed here.
# For development, you might uncomment the below, but for our current setup, leave it commented or use pass.
pass
