# app.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from pathlib import Path

# ----------------------------------------------------------
# 1️⃣ Setup & Load Model
# ----------------------------------------------------------
st.set_page_config(
    page_title="🧬 AI Disease Prediction App",
    page_icon="💊",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = Path("model.pkl")
    if not model_path.exists():
        st.error("❌ Model file 'model.pkl' not found. Please train your model first.")
        st.stop()
    return pickle.load(open(model_path, "rb"))

@st.cache_data
def load_data():
    df = pd.read_csv("disease.csv")
    desc_df = pd.read_csv("symptom_Description.csv") if Path("symptom_Description.csv").exists() else pd.DataFrame()
    prec_df = pd.read_csv("symptom_precaution.csv") if Path("symptom_precaution.csv").exists() else pd.DataFrame()
    return df, desc_df, prec_df

model = load_model()
df, desc_df, prec_df = load_data()

# Determine symptom columns
symptoms = [col for col in df.columns if col not in ['Disease', 'prognosis']]

# ----------------------------------------------------------
# 2️⃣ Title & Info
# ----------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2C3E50;'>🧬 AI Disease Prediction App</h1>
    <p style='text-align:center; font-size:16px; color:#555;'>
    Select your symptoms to get top disease predictions, confidence scores,
    and recommended precautions powered by Machine Learning.
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ App Controls")
st.sidebar.info(
    """
    This AI model predicts the most likely diseases based on selected symptoms.
    
    ⚠️ **Disclaimer:** This tool is for *educational purposes only* and **not a medical diagnosis**.
    """
)

# ----------------------------------------------------------
# 3️⃣ User Input
# ----------------------------------------------------------
st.markdown("### 🧩 Select Symptoms")
selected_symptoms = st.multiselect("Choose from the list below:", sorted(symptoms))

input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

# ----------------------------------------------------------
# 4️⃣ Prediction
# ----------------------------------------------------------
if st.button("🔍 Predict Disease", use_container_width=True):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        try:
            probs = model.predict_proba([input_data])[0]
            classes = model.classes_

            results_df = (
                pd.DataFrame({"Disease": classes, "Confidence": probs})
                .sort_values(by="Confidence", ascending=False)
                .head(3)
                .reset_index(drop=True)
            )

            st.success("✅ Prediction Successful!")
            st.markdown("### 🩺 Top 3 Possible Diseases")

            # ✅ Add interactive confidence bar chart
            fig = px.bar(
                results_df,
                x="Confidence",
                y="Disease",
                orientation="h",
                color="Confidence",
                color_continuous_scale="teal",
                text=results_df["Confidence"].apply(lambda x: f"{x*100:.2f}%"),
                title="Prediction Confidence Levels",
            )
            fig.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Predicted Disease",
                coloraxis_showscale=False,
                template="plotly_white",
                height=350,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top disease details
            top_disease = results_df.iloc[0]['Disease']
            st.markdown(f"## 🧠 Most Likely: **{top_disease}**")

            # Disease Description
            if not desc_df.empty and 'Disease' in desc_df.columns:
                desc_row = desc_df[desc_df['Disease'].str.lower() == top_disease.lower()]
                if not desc_row.empty:
                    st.info(f"🧾 **About:** {desc_row['Description'].values[0]}")
                else:
                    st.info("No description found for this disease.")
            else:
                st.info("Description dataset not loaded.")

            # Precautions
            if not prec_df.empty and 'Disease' in prec_df.columns:
                prec_row = prec_df[prec_df['Disease'].str.lower() == top_disease.lower()]
                if not prec_row.empty:
                    st.markdown("### 🛡️ Recommended Precautions:")
                    for p in prec_row.values.tolist()[0][1:]:
                        if isinstance(p, str) and p.strip():
                            st.markdown(f"- {p}")
                else:
                    st.write("No precaution data found for this disease.")
            else:
                st.write("Precaution dataset not loaded.")

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")

# ----------------------------------------------------------
# 5️⃣ Footer
# ----------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ using <b>Streamlit</b> & <b>Machine Learning</b></p>",
    unsafe_allow_html=True
)
