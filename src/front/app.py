import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

st.set_page_config(page_title="ML Factory - Vitrine", page_icon="🏢")

st.title("🏢 ML Factory - Showcase")

# Config
API_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000") # We reuse the var or use default
# Actually the prompt says the Front must know the API endpoint. Let's provide a setting.
api_endpoint = st.sidebar.text_input("API Endpoint", "http://localhost:8000")

# Header with Version Badge
try:
    health = requests.get(f"{api_endpoint}/health").json()
    version = health.get("model_version", "Discovery")
    st.sidebar.success(f"🟢 Modèle en ligne : Version {version}")
except:
    st.sidebar.error("🔴 API Hors-ligne")
    version = "None"

st.markdown(f"### 🎯 Inférence en temps réel (Version : {version})")

# Data Loading
try:
    test_data = pd.read_csv("data/iris_test.csv")
    st.write("Chargement de `iris_test.csv` réussi.")
    
    selected_index = st.selectbox("Choisir une ligne de test", test_data.index)
    sample = test_data.iloc[selected_index]
    
    features = sample.drop("target").tolist()
    true_label = int(sample["target"])
    
    st.write("**Features sélectionnées :**", features)
    st.write(f"**Vraie étiquette (Ground Truth) :** {true_label}")
    
except Exception as e:
    st.error(f"Fichier de test introuvable : {e}")
    features = [5.1, 3.5, 1.4, 0.2]

if st.button("Lancer l'Inférence"):
    response = requests.post(f"{api_endpoint}/predict", json={"features": features})
    
    if response.status_code == 200:
        res = response.json()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prédiction", res["class_name"])
            st.write(f"ID Version : `{res['model_version']}`")
            
        with col2:
            if res["probabilities"]:
                st.write("**Probabilités :**")
                probs_df = pd.DataFrame({
                    "Espèce": ["Setosa", "Versicolor", "Virginica"],
                    "Confiance": res["probabilities"]
                })
                st.bar_chart(probs_df.set_index("Espèce"))
            else:
                st.warning("Le modèle ne supporte pas le calcul de probabilités.")
                
        if res["prediction"] == true_label:
            st.balloons()
            st.success("✅ Prédiction Correcte !")
        else:
            st.error("❌ Prédiction Incorrecte")
            
    else:
        st.error(f"Erreur API : {response.text}")

st.divider()
st.caption("Infrastructure ML Factory | Zero-Downtime Deployment Demo")
