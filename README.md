# 🏢 ML Factory - Infrastructure "Zero-Downtime"

Bienvenue dans la **ML Factory**. Ce projet démontre comment construire une infrastructure MLOps où le modèle de Machine Learning est totalement découplé de l'application qui le consomme, permettant des mises à jour en temps réel sans interruption de service.

## 🎯 Vision du Projet

L'objectif est d'atteindre une disponibilité de type **Zero-Downtime**. Grâce à l'utilisation d'un **Model Registry** (MLflow) et d'un **Object Storage** (MinIO/S3), vous pouvez mettre à jour l'intelligence de votre API sans jamais redémarrer un seul conteneur.

## 🏗️ Architecture du Workspace

Le projet repose sur une isolation stricte des services :

* **`src/train/`** : Le **Laboratoire**. Script `train.py` pour entraîner et publier des modèles vers le registre (lancé hors Docker pour plus de flexibilité).
* **`src/api/`** : L'**Usine**. API FastAPI qui sert les prédictions en interrogeant dynamiquement MLflow. (Containerisée)
* **`src/front/`** : La **Vitrine**. Interface Streamlit pour tester les modèles avec des données réelles. (Containerisée)
* **Infrastructure** : Pilotée par `docker-compose.yml` incluant **MLflow 2.20.2** (le catalogue) et **MinIO** (le hangar S3).

```text
ml-factory/
├── data/ iris_test.csv    # Données pour les tests de la vitrine
├── src/
│   ├── api/               # FastAPI + model_loader dynamique + Dockerfile
│   ├── front/             # Streamlit + Dockerfile
│   └── train/             # Script d'entraînement (Logistic / Forest)
├── .env                   # Configuration centralisée (S3, MLflow, API)
├── docker-compose.yml     # Orchestration de toute l'usine
└── pyproject.toml         # Gestion des dépendances avec uv
```

---

## 🚀 Guide de Démarrage Rapide

### 1. Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et lancé.
- [uv](https://docs.astral.sh/uv/) (recommandé) ou Python 3.11+.

### 2. Lancer l'Usine (Infrastructure + API + Front)

Utilisez Docker Compose pour tout monter en une seule commande :

```bash
docker-compose up -d --build
```

### 3. URLs d'Accès

- 🌸 **Vitrine (Front)** : [http://localhost:8501](http://localhost:8501)
- ⚙️ **Factory (API)** : [http://localhost:8000/docs](http://localhost:8000/docs)
- 🧪 **Laboratoire (MLflow)** : [http://localhost:5000](http://localhost:5000)
- 📦 **Hangar (MinIO)** : [http://localhost:9001](http://localhost:9001) (Logs: `minioadmin` / `minioadmin`)

---

## 🧪 Scénario de Validation

### Phase 1 : L'Automation (Régression Logistique)

Entraînez et passez automatiquement en production :

```bash
uv run python src/train/train.py --model logistic --auto-publish
```

*Vérifiez sur la Vitrine : la **Version 1** (Logistic Regression) est active.*

### Phase 2 : Le Choix du Chef (Random Forest)

Entraînez un modèle plus complexe sans le publier immédiatement :

```bash
uv run python src/train/train.py --model forest
```

*Vérifiez : La Vitrine utilise toujours la Version 1 (Zero-Downtime).*

### Phase 3 : Bascule Manuelle

1. Allez sur l'interface **MLflow** ([localhost:5000](http://localhost:5000)).
2. Dans **Models** > **IrisClassifier**, sélectionnez la **Version 2**.
3. Ajoutez l'alias **`Production`** à cette version.
4. **Observez** : La Vitrine Streamlit bascule instantanément vers la **Version 2** sans aucun redémarrage !

---

## 🏗️ Comprendre le déploiement Zero-Downtime

Le projet utilise un découplage total entre le stockage et le pilotage pour garantir la portabilité et la sécurité.

### 1. Le trio MLOps

* **MinIO (Le Hangar)** : Stockage physique (S3) des fichiers `.pkl`.
* **MLflow 2.20.2 (Le Chef d'Orchestre)** : Gère le **Registre de Modèles** et les **Aliases**.
* **FastAPI (L'Usine)** : L'API pointe vers l'alias `models:/IrisClassifier@Production`.

### 2. Sécurité et Portabilité

- **Réseau Isolé** : Tous les services communiquent sur un réseau Docker privé (`ml-factory-net`).
- **Pointeurs Dynamiques** : Changer de modèle revient à déplacer un alias dans MLflow, sans toucher au code ni redémarrer les containers.
- **Support des Hôtes** : L'utilisation de MLflow 2.20.2 assure une meilleure compatibilité avec les configurations réseau modernes.

---

## 🛠️ Commandes Utiles

**Voir les logs :**

```bash
docker-compose logs -f api   # Voir les requêtes d'inférence
docker-compose logs -f front # Voir les logs du front
```

**Nettoyer l'infrastructure :**

```bash
docker-compose down -v
```

---

*Projet ML Factory | MLOps Fundamentals*
