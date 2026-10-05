import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import NUMERIC_FEATURES as TRAIN_NUMERIC_FEATURES
from train_model import CATEGORICAL_FEATURES as TRAIN_CATEGORICAL_FEATURES
from train_model import TARGET, train_and_save, validate_dataframe

MODEL_PATH = "model.pkl"
DATA_PATH = "student_mental_health_data.csv"
METRICS_PATH = "metrics.json"
FI_PATH = "feature_importance.csv"
DEFAULT_MODEL_PATH = "model_default.pkl"
DEFAULT_DATA_PATH = "student_mental_health_data_default.csv"
DEFAULT_METRICS_PATH = "metrics_default.json"
DEFAULT_FI_PATH = "feature_importance_default.csv"

# --------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC_FEATURES = [
    "age",
    "daily_social_media_hours",
    "num_platforms_used",
    "sleep_hours",
    "physical_activity_hours",
    "screen_time_before_bed_min",
    "study_hours",
    "face_to_face_hours",
    "academic_performance",
    "social_comparison_score",
    "cyberbullying_experience",
]
CATEGORICAL_FEATURES = ["gender"]


# --------------------------------------------------------------------------------
# Cached resource loading
# --------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("student_mental_health_data.csv")


@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    return pd.read_csv("feature_importance.csv")


model = load_model()
df = load_data()
metrics = load_metrics()
fi_df = load_feature_importance()


def score_band(score):
    if score >= 80:
        return "Thriving", "#2ecc71"
    elif score >= 65:
        return "Doing Well", "#8bc34a"
    elif score >= 50:
        return "Struggling a Bit", "#f1c40f"
    elif score >= 35:
        return "At Risk", "#e67e22"
    else:
        return "Needs Support", "#e74c3c"


# --------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------
st.sidebar.title("🧠 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict My Score", "📊 Data Insights", "🌲 Model Details",
     "📤 Upload & Retrain", "ℹ️ About"],
)

st.sidebar.markdown("---")
with open(DATA_PATH, "rb") as f:
    st.sidebar.download_button(
        "⬇️ Download current dataset (.csv)", f, file_name="student_mental_health_data.csv",
        mime="text/csv", use_container_width=True,
    )
with open(MODEL_PATH, "rb") as f:
    st.sidebar.download_button(
        "⬇️ Download current model (.pkl)", f, file_name="model.pkl",
        mime="application/octet-stream", use_container_width=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ This is an educational demo trained on **synthetic data**. "
    "It is not a clinical or diagnostic tool. If you're struggling, please "
    "reach out to a counselor, trusted adult, or a mental health helpline."
)

# ================================================================================
# PAGE 1 — PREDICT
# ================================================================================
if page == "🔮 Predict My Score":
    st.title("🧠 Predict Your Mental Health Score")
    st.markdown(
        "Adjust the sliders in the sidebar-style panel below to reflect your "
        "daily habits, then hit **Predict** to see your estimated mental "
        "wellbeing score, powered by a Random Forest model."
    )

    col_input, col_result = st.columns([1.1, 1])

    with col_input:
        st.subheader("Your Daily Habits")

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 13, 24, 18)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            daily_social_media_hours = st.slider(
                "Daily social media use (hours)", 0.0, 12.0, 3.5, 0.5
            )
            num_platforms_used = st.slider("Number of platforms used", 0, 9, 3)
            sleep_hours = st.slider("Average sleep per night (hours)", 2.0, 11.0, 7.0, 0.5)
            physical_activity_hours = st.slider(
                "Physical activity / exercise per day (hours)", 0.0, 5.0, 0.8, 0.1
            )
        with c2:
            screen_time_before_bed_min = st.slider(
                "Screen time before bed (minutes)", 0, 180, 45, 5
            )
            study_hours = st.slider("Study hours per day", 0.0, 10.0, 3.0, 0.5)
            face_to_face_hours = st.slider(
                "Face-to-face social time per day (hours)", 0.0, 8.0, 1.5, 0.5
            )
            academic_performance = st.slider("Academic performance (%)", 30, 100, 72)
            social_comparison_score = st.slider(
                "Social comparison tendency (1 = never compare, 10 = constantly)",
                1, 10, 5,
            )
            cyberbullying_experience = st.toggle(
                "Experienced cyberbullying / online harassment?", value=False
            )

        predict_btn = st.button("🔮 Predict My Score", type="primary", use_container_width=True)

    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "daily_social_media_hours": daily_social_media_hours,
        "num_platforms_used": num_platforms_used,
        "sleep_hours": sleep_hours,
        "physical_activity_hours": physical_activity_hours,
        "screen_time_before_bed_min": screen_time_before_bed_min,
        "study_hours": study_hours,
        "face_to_face_hours": face_to_face_hours,
        "academic_performance": academic_performance,
        "social_comparison_score": social_comparison_score,
        "cyberbullying_experience": int(cyberbullying_experience),
    }])

    with col_result:
        st.subheader("Your Result")
        if predict_btn:
            pred = float(model.predict(input_df)[0])
            pred = float(np.clip(pred, 0, 100))
            band, color = score_band(pred)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"suffix": " / 100", "font": {"size": 40}},
                title={"text": f"<b>{band}</b>", "font": {"size": 22, "color": color}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 35], "color": "#fdecea"},
                        {"range": [35, 50], "color": "#fdf3e3"},
                        {"range": [50, 65], "color": "#fdfbe3"},
                        {"range": [65, 80], "color": "#eef8e6"},
                        {"range": [80, 100], "color": "#e3f7ea"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=320, margin=dict(t=60, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            avg_score = df["mental_health_score"].mean()
            delta = pred - avg_score
            st.metric(
                "Compared to average student in dataset",
                f"{pred:.1f}",
                f"{delta:+.1f} vs avg ({avg_score:.1f})",
            )

            # Radar chart: user's normalized inputs vs dataset average
            radar_features = [
                "daily_social_media_hours", "sleep_hours", "physical_activity_hours",
                "face_to_face_hours", "study_hours", "social_comparison_score",
            ]
            radar_labels = [
                "Social Media Use", "Sleep", "Physical Activity",
                "Face-to-Face Time", "Study Hours", "Social Comparison",
            ]
            maxvals = df[radar_features].quantile(0.95)
            user_vals = [input_df[f].iloc[0] / maxvals[f] * 100 for f in radar_features]
            avg_vals = [df[f].mean() / maxvals[f] * 100 for f in radar_features]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_vals + [avg_vals[0]], theta=radar_labels + [radar_labels[0]],
                name="Dataset Average", fill="toself", opacity=0.4,
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=user_vals + [user_vals[0]], theta=radar_labels + [radar_labels[0]],
                name="You", fill="toself", opacity=0.55,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True, height=380, margin=dict(t=30, b=10),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Personalized tips
            st.subheader("💡 Personalized Suggestions")
            tips = []
            if sleep_hours < 7:
                tips.append("😴 Try to get closer to **7–9 hours** of sleep — it's one of the strongest predictors of wellbeing.")
            if daily_social_media_hours > 4:
                tips.append("📱 Consider trimming social media use below **~3 hours/day**; heavy use correlates with lower scores.")
            if screen_time_before_bed_min > 60:
                tips.append("🌙 Cut screen time before bed to under an hour to improve sleep quality.")
            if physical_activity_hours < 0.5:
                tips.append("🏃 Even 20–30 minutes of daily physical activity can meaningfully boost mood.")
            if face_to_face_hours < 1:
                tips.append("🤝 Try to fit in more in-person time with friends or family — it's strongly protective.")
            if social_comparison_score >= 7:
                tips.append("🔍 Notice when you're comparing yourself to others online — consider curating your feed or taking breaks.")
            if cyberbullying_experience:
                tips.append("🆘 If you're experiencing online harassment, please talk to a trusted adult or counselor — you don't have to handle it alone.")
            if not tips:
                tips.append("✅ Your habits look well-balanced — keep it up!")
            for t in tips:
                st.markdown(f"- {t}")
        else:
            st.info("👈 Set your habits and click **Predict My Score** to see your results.")

# ================================================================================
# PAGE 2 — DATA INSIGHTS
# ================================================================================
elif page == "📊 Data Insights":
    st.title("📊 Dataset & Correlation Insights")
    st.markdown(
        f"Exploring the training dataset (**{len(df):,} synthetic student records**) "
        "used to train the model."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records", f"{len(df):,}")
    k2.metric("Avg Score", f"{df['mental_health_score'].mean():.1f}")
    k3.metric("Avg Social Media Use", f"{df['daily_social_media_hours'].mean():.1f} hrs")
    k4.metric("Avg Sleep", f"{df['sleep_hours'].mean():.1f} hrs")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(
            df, x="mental_health_score", nbins=30,
            title="Distribution of Mental Health Scores",
            color_discrete_sequence=["#4c78a8"],
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig3 = px.box(
            df, x="cyberbullying_experience", y="mental_health_score",
            title="Score by Cyberbullying Experience (0=No, 1=Yes)",
            color="cyberbullying_experience",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            df.sample(600, random_state=1), x="daily_social_media_hours",
            y="mental_health_score", color="sleep_hours",
            title="Social Media Use vs Mental Health Score (colored by Sleep)",
            color_continuous_scale="RdYlGn", opacity=0.7,
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig4 = px.scatter(
            df.sample(600, random_state=2), x="sleep_hours",
            y="mental_health_score", color="physical_activity_hours",
            title="Sleep vs Mental Health Score (colored by Activity)",
            color_continuous_scale="Viridis", opacity=0.7,
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    corr_cols = NUMERIC_FEATURES + ["mental_health_score"]
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Feature Correlation Matrix",
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("🔍 Peek at raw data"):
        st.dataframe(df.head(50), use_container_width=True)

# ================================================================================
# PAGE 3 — MODEL DETAILS
# ================================================================================
elif page == "🌲 Model Details":
    st.title("🌲 Model Details & Performance")

    st.markdown(
        "**Algorithm:** Random Forest Regressor (300 trees) inside a scikit-learn "
        "`Pipeline` (one-hot encoding for gender + passthrough numeric features)."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", metrics["R2"])
    m2.metric("MAE", metrics["MAE"])
    m3.metric("RMSE", metrics["RMSE"])
    st.caption(
        f"Evaluated on a held-out test set of {metrics['n_test']} students "
        f"(trained on {metrics['n_train']})."
    )

    st.markdown("---")
    st.subheader("Feature Importance")
    fig_fi = px.bar(
        fi_df.sort_values("importance"), x="importance", y="feature",
        orientation="h", title="What drives the prediction most?",
        color="importance", color_continuous_scale="Blues",
    )
    fig_fi.update_layout(height=450)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(
        "**How to read this:** the Random Forest learns non-linear "
        "interactions between habits — e.g. `cyberbullying_experience`, "
        "`social_comparison_score`, and `sleep_hours` tend to matter most "
        "for predicting a student's mental wellbeing in this dataset."
    )

# ================================================================================
# PAGE 4 — UPLOAD & RETRAIN (backend: bring your own dataset)
# ================================================================================
elif page == "📤 Upload & Retrain":
    st.title("📤 Share Your Own Dataset & Retrain the Model")
    st.markdown(
        "Upload a CSV of real (or your own synthetic) student records and "
        "retrain the model on it right here — no separate server needed. "
        "This acts as the app's lightweight **backend**: validate → train → "
        "save → swap in, all inside this Streamlit session."
    )

    with st.expander("📋 Required CSV format", expanded=False):
        req_cols = TRAIN_NUMERIC_FEATURES + TRAIN_CATEGORICAL_FEATURES + [TARGET]
        st.markdown("Your CSV must contain these **{}** columns:".format(len(req_cols)))
        st.code(", ".join(req_cols))
        st.markdown(
            "- `gender` must be one of: `Male`, `Female`, `Other`\n"
            "- `cyberbullying_experience` must be `0` (No) or `1` (Yes)\n"
            "- `mental_health_score` (target) must be on a **0–100** scale\n"
            "- All other columns must be numeric\n"
            "- At least **30 rows** are required to train"
        )
        template = df[req_cols].head(5)
        st.download_button(
            "⬇️ Download a sample template (5 rows)",
            template.to_csv(index=False),
            file_name="dataset_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that file as CSV: {e}")
            new_df = None

        if new_df is not None:
            st.subheader("Preview")
            st.dataframe(new_df.head(10), use_container_width=True)
            st.caption(f"{len(new_df):,} rows × {new_df.shape[1]} columns")

            is_valid, msg = validate_dataframe(new_df)
            if not is_valid:
                st.error(f"❌ Validation failed: {msg}")
            else:
                st.success(f"✅ {msg} Ready to train on {len(new_df):,} rows.")

                if st.button("🚀 Retrain Model on This Dataset", type="primary"):
                    with st.spinner("Training Random Forest on your data..."):
                        old_metrics = metrics.copy()
                        new_pipe, new_metrics, new_fi = train_and_save(
                            new_df,
                            model_path=MODEL_PATH,
                            data_path=DATA_PATH,
                            metrics_path=METRICS_PATH,
                            fi_path=FI_PATH,
                        )
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("🎉 Model retrained and saved as the active model!")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", new_metrics["R2"], f"{new_metrics['R2'] - old_metrics['R2']:+.3f}")
                    c2.metric("MAE", new_metrics["MAE"], f"{new_metrics['MAE'] - old_metrics['MAE']:+.2f}")
                    c3.metric("RMSE", new_metrics["RMSE"], f"{new_metrics['RMSE'] - old_metrics['RMSE']:+.2f}")
                    st.info(
                        "Go to **Predict My Score** or **Data Insights** — they now "
                        "reflect your uploaded dataset. Use the sidebar download "
                        "buttons any time to export the current model/dataset."
                    )
                    st.caption(
                        "⚠️ Note: on Streamlit Community Cloud, storage resets when "
                        "the app restarts/sleeps — download your retrained model.pkl "
                        "if you want to keep it permanently."
                    )

    st.markdown("---")
    st.subheader("↩️ Reset to Original Synthetic Dataset")
    st.caption("Discard any retrained model and restore the original demo data/model.")
    if st.button("Reset to Default"):
        for src, dst in [
            (DEFAULT_MODEL_PATH, MODEL_PATH),
            (DEFAULT_DATA_PATH, DATA_PATH),
            (DEFAULT_METRICS_PATH, METRICS_PATH),
            (DEFAULT_FI_PATH, FI_PATH),
        ]:
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Restored the original synthetic dataset and model.")

# ================================================================================
# PAGE 5 — ABOUT
# ================================================================================
else:
    st.title("ℹ️ About This Project")
    st.markdown(
        """
### 🎯 What this app does
This project predicts a **student's mental health score (0–100)** from
social media usage and lifestyle habits (sleep, physical activity,
face-to-face interaction, screen time before bed, academic load, social
comparison tendency, and cyberbullying experience) using a **Random Forest
Regression model** trained with scikit-learn.

### 📤 Bring your own data
The **Upload & Retrain** page lets you upload a real (or your own synthetic)
CSV dataset in the required format, validates it, retrains the model on it
live inside the app, and swaps it in as the active model — acting as a
simple, self-contained backend without needing a separate API server. You
can download the current dataset/model at any time from the sidebar, or
reset back to the original demo data.

### 🧪 About the data
The model is trained on a **synthetically generated dataset** of 3,000
students, built with realistic, research-informed relationships (e.g. poor
sleep, excessive night-time screen time, heavy social comparison, and
cyberbullying reduce wellbeing; sleep, exercise, and in-person social time
improve it) plus random noise. It is **not real survey data**, so treat
predictions as illustrative rather than clinical.

### ⚙️ Tech stack
- **scikit-learn** — RandomForestRegressor inside a preprocessing Pipeline
- **pandas / numpy** — data generation & manipulation
- **Plotly** — interactive gauge, radar, and correlation charts
- **Streamlit** — interactive web app & deployment

### 🚀 How to run locally
```bash
pip install -r requirements.txt
python generate_data.py      # (optional) regenerate the dataset
python train_model.py        # (optional) retrain the model
streamlit run app.py
```

### ☁️ How to deploy on Streamlit Community Cloud
1. Push this project folder to a public GitHub repo (include `model.pkl`,
   `student_mental_health_data.csv`, `feature_importance.csv`,
   `metrics.json`, `app.py`, `requirements.txt`).
2. Go to **share.streamlit.io** → **New app**.
3. Select your repo, branch, and set the main file to `app.py`.
4. Click **Deploy** — done!

### 🆘 A note on wellbeing
If you or someone you know is struggling with mental health, please talk to
a school counselor, trusted adult, doctor, or a mental health helpline in
your country. This app is an educational ML demo, not medical advice.
"""
    )
