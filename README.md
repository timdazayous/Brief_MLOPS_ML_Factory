# ML Factory with Intelligent AI Router

Une plateforme MLOps locale complète pour l'entraînement, le suivi et le déploiement de modèles de Machine Learning, enrichie d'un routeur métier intelligent pour intégrer des modèles de langage (LLMs) (Claude, GPT, Mistral, Ollama) de manière optimisée.

## 🎯 Architecture du Projet

Le projet repose sur trois piliers principaux :

1.  **Infrastructure MLOps (Phase 1)** :
    *   **MinIO** : Serveur de stockage objet compatible S3 pour stocker les artefacts de modèles.
    *   **MLflow 2.17.2** : Serveur de tracking d'expériences et de registre de modèles avec une base SQLite et MinIO comme backend d'artefacts.

2.  **Pipeline Machine Learning (Phase 2)** :
    *   Un script `train_register.py` qui entraîne un modèle de classification (Random Forest sur le dataset Iris).
    *   Log automatique des paramètres et métriques dans MLflow.
    *   Enregistrement explicite du modèle dans le Model Registry MLflow et assignation de l'alias `@Production`.

3.  **FastAPI & Routeur IA Intelligent (Phase 3)** :
    *   Une API REST avec FastAPI (`api/main.py`) pour servir les prédictions du modèle en production.
    *   Un chargeur de modèle dynamique avec cache (`api/model_loader.py`) qui récupère toujours la dernière version aliasée `@Production` sans redémarrage.
    *   Un **Routeur IA** (`router/`) qui intercepte des prompts et les redirige vers le LLM le plus adapté en fonction de la compétence requise, du budget restant, et de la confidentialité des données.

---

## 🧠 Le Routeur IA Intelligent (`router/`)

Le routeur IA est conçu pour optimiser les coûts et l'efficacité des appels aux LLMs au sein de la ML Factory. Ses composants sont :

*   **`skill_classifier.py`** : Analyse le prompt (via des mots-clés ou un appel LLM léger) pour classifier la tâche : `REASONING` (Architecture), `CODE` (Génération), `VALIDATION` (Résumé/Parse logs) ou `SENSITIVE` (Données privées).
*   **`token_budget.py`** : Suit la consommation et calcule le coût estimé (en USD) de chaque modèle. Si le budget tombe en dessous de 20%, il déclenche l'utilisation d'un modèle moins coûteux.
*   **`context_compressor.py`** : Si un prompt dépasse une limite de tokens (ex: 2000), ce module le compresse (troncature ou résumé LLM) tout en préservant les données vitales comme les métriques et les erreurs.
*   **`ai_router.py`** : L'orchestrateur principal. Il classe la tâche, vérifie le budget, compresse si besoin, puis appelle le modèle cible. Il inclut une **chaîne de repli (fallback)** (ex: si `claude-opus` échoue, il tente `claude-sonnet`, puis `gpt-4o`, etc., jusqu'à un modèle local `ollama`).

### Matrice de compétences

| Tâche | Modèle préféré | Fallback |
|---|---|---|
| Architecture, raisonnement complexe | claude-opus / gpt-4o | claude-sonnet |
| Génération de code, data wrangling | claude-sonnet / gpt-4o-mini | haiku |
| Validation, résumé, parsing logs | claude-haiku / mistral-small | local ollama |
| Données sensibles / hors-ligne | ollama (llama3 local) | — |

---

## 🛠️ Structure du Projet

```
ml-factory/
├── docker-compose.yml       # Déploiement MinIO et MLflow 2.17.2
├── init_minio.py            # Initialisation boto3 du bucket S3
├── train_register.py        # Pipeline d'entraînement Iris + Model Registry
├── pyproject.toml           # Dépendances Python (compatible uv)
├── requirements.txt         # Dépendances Python (compatible pip)
├── api/
│   ├── main.py              # Serveur FastAPI pour l'inférence
│   └── model_loader.py      # Chargement du modèle @Production avec cache
└── router/
    ├── ai_router.py         # Orchestrateur LLM avec fallback graceful
    ├── skill_classifier.py  # Classification de tâche (REASONING, CODE…)
    ├── token_budget.py      # Suivi financier des tokens consommés
    └── context_compressor.py# Réduction de prompts trop longs
```

---

## 🚀 Guide d'Installation et d'Utilisation

### Prérequis
*   **Docker & Docker Compose**
*   **Python 3.10+**
*   **[uv](https://docs.astral.sh/uv/)** (recommandé) ou `pip`

### Étape 1 : Initialiser l'Infrastructure

Lancez MinIO et MLflow via Docker :

```bash
docker-compose up -d
```

Les services seront disponibles sur :
*   **MinIO Console** (Interface graphique) : [http://localhost:9001](http://localhost:9001)
    *   **Identifiant** : `minioadmin`
    *   **Mot de passe** : `minioadmin`
    *   *Usage : Naviguer dans le bucket `mlflow` pour voir les fichiers `.pkl` et `MLmodel`.*
*   **MLflow UI** : [http://localhost:5000](http://localhost:5000)
    *   *Usage : Suivi des métriques, paramètres et registre de modèles.*

> **⚠️ Première installation ?** Si vous migrez depuis une ancienne version de MLflow,
> pensez à supprimer les anciens volumes Docker pour éviter les erreurs de migration :
> `docker-compose down -v` puis `docker-compose up -d`

### Étape 2 : Installer les dépendances et initialiser MinIO

**Avec `uv` (recommandé)** — utilise `pyproject.toml` :
```bash
uv sync
uv run python init_minio.py
```

**Avec `pip`** — utilise `requirements.txt` :
```bash
pip install -r requirements.txt
python init_minio.py
```

### Étape 3 : Entraîner et Enregistrer le Modèle

Le script `train_register.py` va :
1.  Demander conseil au **Routeur IA** sur les hyperparamètres (Tâche : `REASONING`).
2.  Entraîner un Random Forest sur le jeu de données Iris.
3.  Logger le modèle dans MinIO, l'enregistrer dans le **Model Registry** de MLflow et appliquer l'alias `@Production`.

```bash
uv run python train_register.py
```

> Vérifiez dans MLflow ([http://localhost:5000](http://localhost:5000)) que le modèle `IrisClassifier` version 1 est enregistré avec l'alias `@Production`.

### Étape 4 : Lancer l'API d'Inférence FastAPI

Démarrez le serveur FastAPI. Au lancement, il récupère le modèle `@Production` depuis MLflow et le met en cache.

```bash
uv run python -m uvicorn api.main:app --reload
```

L'API est maintenant disponible sur [http://localhost:8000](http://localhost:8000).

### Étape 5 : Lancer l'interface graphique Streamlit (Optionnel)

Pour tester de manière plus visuelle, une interface Streamlit est disponible. Elle se connecte à l'API FastAPI pour afficher les résultats.

```bash
uv run streamlit run streamlit_app.py
```
L'interface sera disponible sur [http://localhost:8501](http://localhost:8501).

### Étape 6 : Tester l'Inférence et le Routeur (PowerShell)

Déclenchez une prédiction. L'**AI Router** est appelé automatiquement (Tâche : `VALIDATION`) pour formuler une explication de la prédiction.

**Sous Windows PowerShell :**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Réponse attendue :
```json
{
  "prediction": 0,
  "class_name": "Setosa",
  "explanation": "[claude-haiku] Response to: The model predicted class 0 (Setosa)...",
  "router_cost_usd": 0.000034,
  "model_version": "1",
  "message": "Prediction successful."
}
```

**Exemples de features à tester :**

| Fleur | Features |
|---|---|
| 🌸 Setosa | `[5.1, 3.5, 1.4, 0.2]` |
| 🌺 Versicolor | `[6.3, 3.3, 4.7, 1.6]` |
| 🌼 Virginica | `[7.2, 3.6, 6.1, 2.5]` |

---

## ⚙️ Notes Techniques

| Composant | Version |
|---|---|
| MLflow (client + serveur) | **2.17.2** |
| MinIO | latest |
| Python | ≥ 3.10 |
| setuptools | ≥ 69, < 72 (requis pour `pkg_resources` utilisé par MLflow) |

> **Pourquoi `setuptools` ?** MLflow importe `pkg_resources` au démarrage. Les environnements créés par `uv` n'incluent pas `setuptools` par défaut, et les versions ≥ 72 ont supprimé ce module. La contrainte `>=69,<72` garantit la compatibilité.

---

## 🔮 Évolutions Futures
*   **Connexions LLM Réelles** : Remplacez la fonction `_mock_call` de `router/ai_router.py` par la bibliothèque [`litellm`](https://github.com/BerriAI/litellm) pour faire de véritables requêtes vers OpenAI, Anthropic ou Ollama en une seule API unifiée.
*   **Compression par LLM** : Remplacez la logique de troncature basique dans `context_compressor.py` par un appel réel à un modèle `haiku` ou `mistral-small`.
*   **UI Gradio/Streamlit** : Ajouter un front-end pour interagir directement avec le routeur et l'API d'inférence de manière plus visuelle.
