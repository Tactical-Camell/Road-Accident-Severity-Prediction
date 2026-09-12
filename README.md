# 🚦 Road Accident Severity Prediction & Geospatial Risk Mapping

A comprehensive machine learning and geospatial analytics framework for analyzing road crash severity, detecting geographic hotspots using **DBSCAN clustering**, computing severity-weighted risk indices, and interactively visualizing accident risks across the **City of Chicago, IL, USA** crash dataset.

---

## 📌 Project Overview

This project combines predictive machine learning models, advanced spatial analytics, and interactive web applications to answer key traffic safety questions:
1. **Severity Prediction**: Classifying and predicting crash severity based on environmental, temporal, road condition, and vehicle attributes.
2. **Geospatial Hotspot Detection**: Identifying spatial accident clusters using **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** with Haversine metrics.
3. **Regional Risk Scoring**: Aggregating severity metrics across spatial boundaries (community areas) and clusters to rank high-risk zones.
4. **Interactive Exploration**: Enabling dynamic data exploration through Plotly, Folium, and multi-featured Dash applications.

---

## 📊 Dataset & DBSCAN Hotspot Clustering

### **Chicago Dataset (Chicago, IL, USA)** — *~1,000,000 Records*
- **Spatial Unit**: 77 Chicago Community Areas (`Boundaries - Community Areas`).
- **Processing**:
  - Full data cleaning pipeline handling missing values, temporal features, and injury categorizations.
  - Spatial join linking crash point geometries with 77 official community area polygons.
  - **Scalable Grouped DBSCAN**: To efficiently scale DBSCAN clustering to ~1,000,000 records without memory bottlenecks, DBSCAN clustering is executed in parallel groups per community area (`group_col="community_area_number"`).
  - Calculated crash-level, community-level, and cluster-level severity risk indices.
- **Visualizations**: Interactive Dash applications (`dashapp.py`, `interactive_dash.py`) with dynamic filtering by community area, minimum crash threshold, and severity categories.

---

## 📈 Weighted Severity Risk Score

To accurately quantify geographic danger beyond raw crash counts, a weighted severity index is computed for each region and DBSCAN cluster:

$$\text{Risk Index} = \frac{3 \times \text{Fatal} + 2 \times \text{Incapacitating/Serious} + 1 \times \text{Non-Incapacitating/Moderate} + 0.5 \times \text{Minor/Possible}}{\text{Total Crashes}}$$

| Severity Level | Weight | Description |
|---|---|---|
| **Fatal Injury** | `3.0` | Fatal crashes |
| **Serious / Incapacitating Injury** | `2.0` | Severe injuries requiring hospitalization |
| **Moderate / Non-Incapacitating Injury** | `1.0` | Evident non-incapacitating injuries |
| **Minor / Possible Injury** | `0.5` | Reported minor injuries / pain |

A higher score indicates a higher concentration of severe or fatal accidents relative to total crash count.

---

## 🧠 Machine Learning & Explainability

- **Model Architectures**: Logistic Regression, Extra Trees, XGBoost, and LightGBM models trained for crash severity classification.
- **Class Imbalance Handling**: Integrated SMOTE (Synthetic Minority Over-sampling Technique) in `imblearn` pipelines.
- **Explainability**: Feature importance analysis and SHAP/LIME outputs (`explainability.py`) to highlight key severity drivers (speed limits, road alignment, weather, lighting conditions).

---

## 🧱 Project Directory Structure

```
Road-Accident-Severity-Prediction/
│
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Python dependencies
│
└── road-accident-severity/
    ├── modelling.py                   # Machine learning model training & cross-validation
    ├── explainability.py              # Model explainability & feature importance analysis
    │
    ├── reports/                       # Confusion matrices, CV comparisons, feature importance plots
    │   ├── confusion_matrix.png
    │   ├── cross validation metrics comparison.png
    │   ├── cv_results.csv
    │   └── feature_importance_LightGBM.png
    │
    └── GeospatialRisk/
        └── Chicago/
            ├── cleaning.py            # Data cleaning pipeline for Chicago crash data
            ├── prepare_geodata.py     # GeoDataFrame conversion & spatial join with 77 community areas
            ├── dbscan_hotspots.py     # Scalable per-community area DBSCAN hotspot clustering
            ├── severity_pipeline.py   # Crash, community area, and cluster risk scoring
            └── app/
                ├── dashapp.py         # Dash choropleth & hotspot web dashboard
                └── interactive_dash.py# Advanced interactive Dash application with live filters
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Chirudeva-Reddy/Road-Accident-Severity-Prediction.git
cd Road-Accident-Severity-Prediction
git checkout geospatial-clustering-chicago
pip install -r requirements.txt
```

### 2. Run Geospatial Risk Pipeline (Chicago)
```bash
# Clean and prepare GeoJSON spatial datasets
python -m road-accident-severity.GeospatialRisk.Chicago.prepare_geodata

# Execute DBSCAN hotspot clustering across community areas
python -m road-accident-severity.GeospatialRisk.Chicago.dbscan_hotspots

# Compute severity risk indices
python -m road-accident-severity.GeospatialRisk.Chicago.severity_pipeline
```

### 3. Launch Interactive Dash App
```bash
python road-accident-severity/GeospatialRisk/Chicago/app/interactive_dash.py
```
Open your browser and navigate to `http://127.0.0.1:8050/`.

---

## 👥 Contributors & License

- Academic project for Road Accident Severity Prediction and Geospatial Risk Analytics.
- Licensed for educational and research use.
