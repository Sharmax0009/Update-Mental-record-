# 🧠 Predicting Student Mental Health Score from Social Media Usage

An interactive AI/ML Streamlit app that predicts a student's mental health
score (0–100) from social media usage and lifestyle habits, using a
Random Forest regression model.

## Features
- 🔮 **Live prediction** with sliders/toggles for 12 lifestyle inputs
- 📟 **Animated gauge chart** showing your predicted score & wellbeing band
- 🕸️ **Radar chart** comparing your habits to the dataset average
- 💡 **Personalized suggestions** based on your inputs
- 📊 **Data Insights page** — histograms, scatter plots, correlation heatmap
- 🌲 **Model Details page** — R², MAE, RMSE, feature importance chart
- 📤 **Upload & Retrain page (backend)** — upload your own CSV dataset,
  validate it, retrain the model live inside the app, compare metrics
  against the previous model, and reset back to the default demo data
  at any time
- ⬇️ **Download buttons** (sidebar) for the current dataset and model file
- ℹ️ **About page** with methodology & deployment guide

## Project structure
```
mental_health_app/
├── app.py                          # Streamlit app (run this)
├── generate_data.py                # Synthetic dataset generator
├── train_model.py                  # Trains & saves the ML model
├── model.pkl                       # ACTIVE trained sklearn Pipeline
├── student_mental_health_data.csv  # ACTIVE dataset
├── feature_importance.csv          # ACTIVE feature importances
├── metrics.json                    # ACTIVE model evaluation metrics
├── model_default.pkl                       # Backup of the original demo model
├── student_mental_health_data_default.csv  # Backup of the original demo dataset
├── feature_importance_default.csv          # Backup of original feature importances
├── metrics_default.json                    # Backup of original metrics
├── requirements.txt
└── README.md
```

## Quick start (local)
```bash
pip install -r requirements.txt

# Optional: regenerate data & retrain (pre-built files already included)
python generate_data.py
python train_model.py

streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a new **public GitHub repo** and push this entire folder to it.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, set main file path to `app.py`.
4. Click **Deploy**. Your app will be live at `https://<your-app>.streamlit.app`.

> Make sure `model.pkl`, `student_mental_health_data.csv`,
> `feature_importance.csv`, and `metrics.json` are committed to the repo —
> the app loads them directly rather than retraining on every run.

## Bringing your own dataset (backend)
Go to the **📤 Upload & Retrain** page in the app to plug in real data:
1. Check the required column list (expand "Required CSV format").
2. Upload a CSV matching that schema (download the sample template if needed).
3. The app validates it (column names, types, row count, value ranges).
4. Click **Retrain Model on This Dataset** — it trains a fresh Random Forest,
   saves it as the active `model.pkl`, and shows you the new R²/MAE/RMSE
   next to the old ones.
5. Every other page (Predict, Data Insights, Model Details) immediately
   reflects the new data/model.
6. Use **Reset to Default** any time to restore the original synthetic demo
   data, or the sidebar **Download** buttons to export the current
   dataset/model.

> **Note:** Streamlit Community Cloud's filesystem is ephemeral — if the app
> restarts or goes to sleep, it reloads the files from your GitHub repo, so
> a retrained model from a live session won't persist unless you download it
> and commit it back to the repo yourself.

## Model
- **Algorithm:** RandomForestRegressor (300 trees, max_depth=10) in a
  scikit-learn Pipeline with one-hot encoding for `gender`.
- **Performance (held-out test set):** R² ≈ 0.75, MAE ≈ 4.5 points (0–100 scale).
- **Top predictive features:** cyberbullying experience, social comparison
  tendency, sleep hours, daily social media hours, face-to-face time.

## Disclaimer
This app is trained on **synthetic data** for educational/demo purposes.
It is **not a diagnostic or clinical tool**. If you or someone you know is
struggling, please reach out to a school counselor, doctor, trusted adult,
or a local mental health helpline.
