# app.py

import streamlit as st
import pandas as pd
import pickle

# ----------------------------------------------------------
# 1️⃣ Load model and datasets
# ----------------------------------------------------------
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('disease.csv')

# Optional supporting datasets
try:
    desc_df = pd.read_csv('symptom_Description.csv')
    prec_df = pd.read_csv('symptom_precaution.csv')
except FileNotFoundError:
    desc_df = pd.DataFrame()
    prec_df = pd.DataFrame()

# Determine symptom columns
symptoms = list(df.columns)
if 'Disease' in symptoms:
    symptoms.remove('Disease')
elif 'prognosis' in symptoms:
    symptoms.remove('prognosis')

# ----------------------------------------------------------
# 2️⃣ Streamlit App Layout
# ----------------------------------------------------------
st.set_page_config(page_title="🧬 AI Disease Prediction App", page_icon="💊", layout="centered")

st.title("🧬 AI Disease Prediction from Symptoms")
st.write("Select your symptoms to get top 3 possible disease predictions with confidence scores, descriptions, and precautions.")

# Sidebar info
st.sidebar.title("About the App")
st.sidebar.info("""
This AI model predicts the most likely diseases based on selected symptoms.
It also provides short descriptions and precautionary steps.

⚠️ For educational purposes only — not medical advice.
""")

# ----------------------------------------------------------
# 3️⃣ User Input
# ----------------------------------------------------------
selected_symptoms = st.multiselect("Select Symptoms", symptoms)

input_data = [0] * len(symptoms)
for symptom in selected_symptoms:
    input_data[symptoms.index(symptom)] = 1

# ----------------------------------------------------------
# 4️⃣ Prediction with Confidence Scores
# ----------------------------------------------------------
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Predict probabilities for all diseases
        probs = model.predict_proba([input_data])[0]
        classes = model.classes_

        # Create DataFrame for top 3 predictions
        results_df = pd.DataFrame({
            "Disease": classes,
            "Confidence": probs
        }).sort_values(by="Confidence", ascending=False).head(3)

        st.subheader("🏥 Top 3 Possible Diseases")
        for i, row in results_df.iterrows():
            st.write(f"**{row['Disease']}** — {row['Confidence']*100:.2f}% confidence")

        # Get top disease
        top_disease = results_df.iloc[0]['Disease']
        st.success(f"🩺 Most Likely Disease: **{top_disease}**")

        # Description
        if not desc_df.empty:
            desc_row = desc_df[desc_df['Disease'] == top_disease]
            if not desc_row.empty:
                st.info(f"**About:** {desc_row['Description'].values[0]}")
            else:
                st.info("No description available.")
        else:
            st.info("Description dataset not found.")

        # Precautions
        if not prec_df.empty:
            prec_row = prec_df[prec_df['Disease'] == top_disease]
            if not prec_row.empty:
                st.markdown("### 🛡️ Recommended Precautions:")
                for p in prec_row.values.tolist()[0][1:]:
                    if isinstance(p, str) and p.strip():
                        st.markdown(f"- {p}")
            else:
                st.write("No precaution data available.")
        else:
            st.write("Precaution dataset not found.")

# ----------------------------------------------------------
# 5️⃣ Footer
# ----------------------------------------------------------
st.markdown("---")
st.info("ℹ️ This prediction is based on patterns in the dataset. "
                "Always consult a certified medical professional for real diagnosis.")
