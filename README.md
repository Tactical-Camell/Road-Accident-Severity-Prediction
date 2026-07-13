# 🚦 NSW Road Accident Severity Prediction & Geospatial Risk Mapping 
---
## 📌 Project Overview
This project analyzes road crash data from New South Wales (NSW), Australia to:

1. **Predict crash severity** using cleaned crash attributes  
2. **Identify high-risk geographic regions** using geospatial analysis  
3. **Detect hotspot clusters** using DBSCAN  
4. **Compute regional severity-weighted risk scores**  
5. **Visualize crash risk interactively** using Plotly, Folium, and Dash  

The project integrates machine learning, geospatial data processing, clustering, and interactive mapping to build a complete analytics pipeline.

---

## 🧱 Project Structure

```
FDS-PROJECT/
│
├── data/
│   ├── cleaned/                     # Cleaned crash dataset (from teammate)
│   ├── raw/                         # Raw shapefiles / auxiliary files
│   └── output/                      # Generated datasets & maps
│
│── step1_prepare_postcodes.py        # Convert shapefile to GeoJSON
├── step2_prepare_crash_points.py     # Convert lat/long → geometry Points
├── step3_reverse_geocode_poa.py      # Spatial join with POA shapefile
├── step4_dbscan_clustering.py        # DBSCAN clustering on crash points
├── step5_compute_risk_scores.py      # ZIP + cluster weighted severity scoring
├── step6_plotly_choropleth.py        # Plotly choropleth map of NSW risk
├── step6_folium_hotspots.py          # Folium DBSCAN hotspot/heatmap visualization
├── step7_dash_app.py                 # Dash interactive crash risk explorer
│
├── README.md                         # (this file)
└── requirements.txt
```

---

## 📊 Data Used

### **Main Crash Dataset (Cleaned)**
Provided by NSW Open Data / TfNSW:
- Latitude / Longitude  
- Crash date, reporting year  
- Severity components (Killed, Serious, Moderate, Minor)  
- Road conditions, lighting, environment  
- Weather  
- Street information  
- Traffic units involved  

### **Geospatial Data**
- **ABS POA 2021 Shapefile**  
  Used for postcode assignment via spatial join  
- Geometry converted to WGS84 (`EPSG:4326`) for mapping  
- Optional simplified geometries for Dash performance  

---

## 🧠 Machine Learning: Severity Prediction  
A separate notebook/model predicts crash severity using:
- road conditions  
- lighting  
- alignment  
- weather  
- traffic units  
- speed limit  
- location characteristics  

(This model work is handled by teammates; geospatial analysis is documented here.)

---

## 🌐 Geospatial Pipeline (Fully Implemented)

### **1️⃣ Convert crash points to geometry**
`step2_prepare_crash_points.py`
- Validates lat/long  
- Creates a GeoDataFrame  
- Saves as GeoJSON

### **2️⃣ Assign crash to a postcode (POA)**
`step3_reverse_geocode_poa.py`
- Spatial join between crash point and POA polygon  
- Adds:
  - `POA_CODE21`
  - `POA_NAME21`

### **3️⃣ Hotspot detection using DBSCAN**
`step4_dbscan_clustering.py`
- Haversine-distance DBSCAN  
- Identifies spatial clusters of crashes  
- Adds:
  - `cluster_id`
  - distances
  - noise points (`cluster_id = -1`)

### **4️⃣ Severity-weighted risk scoring**
`step5_compute_risk_scores.py`
Generates:
- `zip_risk_table.csv`  
- `cluster_risk_table.csv`

Weighted Severity Score:
```
risk = (3*killed + 2*serious + 1*moderate + 0.5*minor) / total_crashes
```

### **5️⃣ Plotly Choropleth (NSW Risk Map)**
`step6_plotly_choropleth.py`
- Smooth, modern Viridis color scale  
- Hover stats  
- Clear severity shading  
- Output: `plotly_nsw_risk_map.html`

### **6️⃣ Folium Hotspot Map**
`step6_folium_hotspots.py`
Layers:
- Raw crashes  
- DBSCAN clusters (colored + scaled by severity)  
- Heatmap  
- NSW postcode outlines  
- LayerControl  
Output:  
`folium_hotspot_map.html`

### **7️⃣ Dash Interactive Application**
`step7_dash_app.py`
Fully interactive crash risk explorer:
- Metric dropdown  
- Minimum crash count slider  
- Color scale selector  
- Live Mapbox choropleth  
Runs at:
```
http://127.0.0.1:8050
```

---

## 📈 Weighted Severity Score (Key Concept)

This score quantifies how dangerous a postcode or cluster is:

| Severity Type        | Weight |
|----------------------|--------|
| Fatal                | 3      |
| Serious Injury       | 2      |
| Moderate Injury      | 1      |
| Minor / Other Injury | 0.5    |

A higher score ⇒ **more severe crashes on average**.

This forms the basis of the regional risk map.

---

## 🚀 How to Run Everything

### **1. Install dependencies**
```
pip install -r requirements.txt
```

### **2. Generate POA assignments**
```
python step3_reverse_geocode_poa.py
```

### **3. Generate DBSCAN clusters**
```
python step4_dbscan_clustering.py
```

### **4. Compute risk tables**
```
python step5_compute_risk_scores.py
```

### **5. Generate Plotly Choropleth**
```
python step6_plotly_choropleth.py
```

### **6. Generate Folium Hotspot Map**
```
python step6_folium_hotspots.py
```

### **7. Run Dash App**
```
python step7_dash_app.py
```

---

## 🧭 Final Outputs

Generated into `data/output/`:

- `crashes_with_postcodes.geojson`
- `crashes_with_clusters.geojson`
- `zip_risk_table.csv`
- `cluster_risk_table.csv`
- `plotly_nsw_risk_map.html`
- `folium_hotspot_map.html`
- Dash web app for interactive exploration

---

## 📝 Notes / Assumptions

- The original crash dataset already includes a `Weather` attribute  
  → No need for external historical weather API queries  
- All geospatial operations use **EPSG:4326**  
- DBSCAN uses haversine distance with tuned epsilon  
- POA boundaries are from **ABS 2021**  

---

## 👥 Team Members
- *Your Names*  
- Roles:  
  - Data Cleaning  
  - Prediction Modeling  
  - Geospatial Analysis  
  - Visualization / Dash  

---

## 📌 License  
For academic use only.