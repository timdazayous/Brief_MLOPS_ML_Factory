import streamlit as st
import requests
import json

st.set_page_config(
    page_title="ML Factory — Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 ML Factory — Iris Classifier")
st.markdown("""
Cette interface communique avec l'API FastAPI pour prédire l'espèce d'une fleur d'Iris et obtenir une explication via le **Routeur IA**.
""")

st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("URL de l'API", "http://localhost:8000")

st.header("📊 Entrées du modèle")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Longueur du sépale (cm)", 4.0, 8.0, 5.1, step=0.1)
    sepal_width = st.number_input("Largeur du sépale (cm)", 2.0, 4.5, 3.5, step=0.1)

with col2:
    petal_length = st.number_input("Longueur du pétale (cm)", 1.0, 7.0, 1.4, step=0.1)
    petal_width = st.number_input("Largeur du pétale (cm)", 0.1, 2.5, 0.2, step=0.1)

if st.button("🚀 Prédire", use_container_width=True):
    payload = {
        "features": [sepal_length, sepal_width, petal_length, petal_width]
    }
    
    try:
        with st.spinner("Appel à l'API..."):
            response = requests.post(f"{api_url}/predict", json=payload)
            response.raise_for_status()
            data = response.json()
        
        st.success(f"### Résultat : {data['class_name']}")
        
        st.info(f"**💡 Explication de l'IA :**\n\n{data['explanation']}")
        
        # Affichage des métriques
        m1, m2, m3 = st.columns(3)
        m1.metric("Classe", data['prediction'])
        m2.metric("Coût Routeur", f"${data['router_cost_usd']:.5f}")
        m3.metric("Version Modèle", data['model_version'])
        
    except requests.exceptions.ConnectionError:
        st.error("Impossible de contacter l'API. Vérifiez que `uvicorn` tourne bien sur le port 8000.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")

st.divider()
st.caption("Projet ML Factory — MLflow / MinIO / FastAPI / Streamlit / AI Router")
