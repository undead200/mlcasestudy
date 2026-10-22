# Machine Learning Capstone Project (23CSE301)

A complete, end-to-end Machine Learning engineering pipeline covering three fundamental tracks: **Regression**, **Classification**, and **Clustering**. Designed and implemented to adhere strictly to the **23CSE301 Capstone Project Guidelines and Evaluation Rubrics**.

---

## 📌 Problem Statements & Assigned Datasets

### 1. 🛵 Regression Track: Food Delivery Time Prediction
* **Dataset**: `food_delivery_regression.csv` (45,593 rows)
* **Objective**: Predict delivery duration `Time_taken(min)` based on driver attributes, geographical coordinates, weather, traffic density, and order specifics.
* **Pipeline Highlights**:
  * Feature Engineering: Haversine distance calculation (`Distance_km`).
  * Stratified target binning (`pd.qcut`) for an 80:20 train/test split.
  * Systematic evaluation of 10 regression algorithms (Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN).
  * 5-fold cross-validation and `GridSearchCV` hyperparameter tuning.

### 2. 💧 Classification Track: Water Quality & Potability Assessment
* **Dataset**: `water_potability_classsification.csv` (3,276 rows)
* **Objective**: Predict water safety for human consumption (`Potability`: 1 = Potable, 0 = Non-Potable) using physical and chemical attributes.
* **Pipeline Highlights**:
  * Feature Engineering: WHO-compliant safety threshold indicator (`is_ph_optimal`).
  * 80:20 stratified train/test split with zero-leakage `StandardScaler` (fitted purely on train).
  * Evaluated across all 10 mandated algorithms: Part A baseline (Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC) and Part B ensembles (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP Classifier).
  * Decision tree structure visualization (`plot_tree`) and feature importance extraction.
  * Consolidated 10-algorithm comparison table (Accuracy, Precision, Recall, F1-weighted, ROC-AUC) and 5-fold cross-validation.
  * Hyperparameter tuning of Random Forest via `GridSearchCV` with documented before-vs-after improvements.

### 3. 📈 Clustering Track: Stock Market Regime Discovery
* **Dataset**: `clustering_dataset.csv` (12,071 rows)
* **Objective**: Unsupervised segmentation of financial market regimes across stock price and liquidity dimensions without label supervision.
* **Pipeline Highlights**:
  * Variance stabilization via $\log(1 + \text{Volume})$ to address extreme heavy-tailed volume distributions.
  * K-Means clustering with Elbow curve analysis ($k=2 \dots 10$) selecting optimal $k=4$.
  * Agglomerative Hierarchical Clustering with Dendrogram visualization and systematic linkage comparison (`ward`, `complete`, `average`).
  * Multi-dimensional manifold visualisations via **2D PCA** (99.98% explained variance) and **2D t-SNE**.
  * Post-hoc ground-truth validation against stock `Ticker` labels using Contingency Matrix, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI) (Rubric Section 3.3).

---

## 🏆 Model Performance Summary (Rubric Section 6, C2)

### Regression Track (Held-Out Test Set)
| Rank | Algorithm | $R^2$ Score | RMSE (min) | MAE (min) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Gradient Boosting Regressor (Tuned)** | **0.8340** | **3.98** | **3.12** | Production Best |
| 2 | Random Forest Regressor | 0.8124 | 4.21 | 3.34 | Ensemble Baseline |
| 3 | Decision Tree Regressor | 0.7410 | 4.95 | 3.91 | Non-linear Tree |
| 4 | KNN Regressor | 0.6231 | 5.92 | 4.67 | Instance-based |
| 5 | SVR (RBF Kernel) | 0.6012 | 6.09 | 4.81 | Kernel Method |
| 6 | Polynomial Regression (deg=2) | 0.5420 | 6.52 | 5.18 | Non-linear Features |
| 7 | Ridge Regression | 0.5218 | 6.67 | 5.29 | Regularized $L_2$ |
| 8 | Linear Regression | 0.5217 | 6.67 | 5.29 | Linear Baseline |
| 9 | Lasso Regression | 0.5209 | 6.68 | 5.30 | Regularized $L_1$ |
| 10 | ElasticNet Regression | 0.5211 | 6.67 | 5.29 | Combined $L_1 + L_2$ |

### Classification Track (Held-Out Test Set)
| Rank | Algorithm | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Random Forest (Tuned)** | **0.6720** | **0.6333** | **0.3203** | **0.6321** | **0.6891** | Production Best |
| 2 | Gradient Boosting | 0.6616 | 0.6061 | 0.3125 | 0.6225 | 0.6780 | Ensemble Boosting |
| 3 | Support Vector Machine (SVC) | 0.6494 | 0.6250 | 0.1953 | 0.5794 | 0.6658 | Kernel Method |
| 4 | Bagging (Decision Tree) | 0.6418 | 0.5574 | 0.3984 | 0.6268 | 0.6672 | Ensemble Bagging |
| 5 | MLP Classifier | 0.6296 | 0.5312 | 0.3984 | 0.6174 | 0.6677 | Neural Network |
| 6 | AdaBoost Classifier | 0.6265 | 0.5342 | 0.3047 | 0.5986 | 0.6335 | Adaptive Boosting |
| 7 | Decision Tree | 0.5976 | 0.4855 | 0.5234 | 0.6021 | 0.5843 | Single Tree |
| 8 | Naive Bayes (Gaussian) | 0.6265 | 0.5625 | 0.1406 | 0.5492 | 0.6166 | Probabilistic |
| 9 | K-Nearest Neighbors | 0.5838 | 0.4578 | 0.4453 | 0.5847 | 0.5658 | Instance-based |
| 10 | Logistic Regression | 0.6098 | 0.0000 | 0.0000 | 0.4616 | 0.5273 | Linear Baseline |

### Clustering Track (Unsupervised Partitioning)
| Algorithm / Linkage | Sample Size ($N$) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=4$)** | **12,071** | **0.4390** | **0.7997** | **13,717.20** | Optimal Partition |
| Agglomerative (Ward) | 1,000 | 0.3802 | 0.8188 | 1,027.57 | Hierarchical Best |
| Agglomerative (Complete) | 1,000 | 0.3341 | 0.9421 | 745.32 | Alternative Linkage |
| Agglomerative (Average) | 1,000 | 0.3129 | 0.9854 | 698.41 | Alternative Linkage |

---

## 📁 Repository Structure (Rubric Section 8)

```text
/
├── README.md                           # Comprehensive documentation & results summary
├── requirements.txt                     # Pinned Python package dependencies
├── data/                               # Raw datasets
│   ├── food_delivery_regression.csv
│   ├── water_potability_classsification.csv
│   └── clustering_dataset.csv
├── notebooks/                          # Clean, fully-executed Jupyter notebooks
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── Clustering.ipynb
├── models/                             # Exported model binaries & preprocessors (.pkl)
│   ├── best_regression_model.pkl
│   ├── regression_preprocessor.pkl
│   ├── regression_meta.pkl
│   ├── best_classification_model.pkl
│   ├── classification_scaler.pkl
│   ├── classification_meta.pkl
│   ├── kmeans_clustering_model.pkl
│   ├── clustering_scaler.pkl
│   └── clustering_meta.pkl
└── app/                                # Interactive Web Application (Bonus Item)
    └── app.py                          # Streamlit multi-track interactive portal
```

---

## 💻 Local Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/undead200/mlcasestudy.git
   cd mlcasestudy
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Jupyter Notebooks:**
   ```bash
   jupyter notebook notebooks/
   ```

---

## 🌐 Interactive Web Application & Live Cloud Deployment

### 📦 Essential Folders & Files Required for Live Deployment
For the live application to run in the cloud, the following files and folders **must be committed and pushed to your GitHub repository**:
1. **`app/`**: Contains [`app/app.py`](file:///app/app.py), the interactive Streamlit web application.
2. **`models/`**: Contains all 9 trained model binaries, scalers, and metadata files:
   - `best_regression_model.pkl` & `regression_preprocessor.pkl` & `regression_meta.pkl`
   - `best_classification_model.pkl` & `classification_scaler.pkl` & `classification_meta.pkl`
   - `kmeans_clustering_model.pkl` & `clustering_scaler.pkl` & `clustering_meta.pkl`
   *(All files are under 11 MB each, well within GitHub's 100 MB limit — no Git LFS needed!)*
3. **`requirements.txt`**: Located in the repository root so the cloud hosting platform automatically installs all required Python dependencies.

---

### 🚀 Step-by-Step Guide: Deploying the App Live (Streamlit Community Cloud)

**Streamlit Community Cloud** is the recommended platform for hosting this project live for free with automatic HTTPS, continuous deployment, and zero server maintenance.

#### Step 1: Push All Necessary Files to GitHub
Open your terminal in the project directory and run:
```bash
# Check status of changed files
git status

# Stage the app, models, requirements, and documentation
git add app/ models/ requirements.txt README.md .gitignore

# Commit changes
git commit -m "Prepare live deployment assets and streamlined regression UI"

# Push to your GitHub repository
git push origin main
```
*(If your default branch is `master`, use `git push origin master`)*

#### Step 2: Sign In to Streamlit Community Cloud
1. Open [share.streamlit.io](https://share.streamlit.io/) in your browser.
2. Click **"Sign in with GitHub"** and authorize Streamlit.

#### Step 3: Configure and Launch Your App
1. On your Streamlit dashboard, click the **"New app"** (or **"Create app"**) button.
2. Fill in the deployment details:
   - **Repository**: Select your GitHub repo (e.g., `your-username/mlcasestudy` or `undead200/mlcasestudy`).
   - **Branch**: `main` (or `master`).
   - **Main file path**: `app/app.py`.
   - **App URL (optional)**: Customize your sub-domain (e.g., `ml-capstone-portal.streamlit.app`).
3. Click **"Deploy!"**.

#### Step 4: Verification & Live Access
* Streamlit Cloud will read `requirements.txt`, install dependencies, load the serialized models from `models/`, and start `app/app.py`.
* In about 1–2 minutes, your web application will be live at a public HTTPS URL (shareable with instructors, evaluators, or in resumes).
* Any subsequent `git push` to your GitHub repo will automatically trigger a live update without downtime.

---

### 🔄 Alternative Free Live Deployment Options

<details>
<summary><b>Option B: Hugging Face Spaces (Click to expand)</b></summary>

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create an account.
2. Click **"Create new Space"**.
3. Choose **Streamlit** as the Space SDK and set the visibility to **Public**.
4. Clone the space repo locally or push your `app/app.py`, `models/`, and `requirements.txt` to the HF Space Git repository.
5. Hugging Face will automatically build the container and provide a live URL.
</details>

<details>
<summary><b>Option C: Render.com (Click to expand)</b></summary>

1. Go to [Render.com](https://render.com/) and link your GitHub account.
2. Click **"New +"** > **"Web Service"**.
3. Select your GitHub repository.
4. Set:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Create Web Service"**.
</details>

---

## ⚖️ Academic Integrity & AI Tools Citation (Rubric 7.5 & 8)

In compliance with Capstone Guidelines Section 7.5:
* **Generative AI Tools**: Utilized for code scaffolding, boilerplate templates, and markdown document structuring.
* **Originality**: All exploratory data analysis, domain-specific feature engineering decisions, hyperparameter tuning grids, and insight interpretations were formulated, tested, and validated by the project team members.
