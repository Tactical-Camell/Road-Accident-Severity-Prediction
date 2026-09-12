"""
explainability.py
Generate SHAP global + local explanations, LIME local explanations,
and an HTML report for the road-accident-severity project.

Place this file in your project root (the folder containing models/ and Dataset/).
Run with: python explainability.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import base64
import shutil
import os
from sklearn.model_selection import train_test_split

# -----------------------
# Paths - use script location as project root (robust across OS)
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = Path("/Users/Dell/Desktop/vscode projects/fds/Dataset/data_labeled.csv")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPLAIN_DIR = REPORTS_DIR / "explainability_outputs"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

# optional: if you have any pre-generated feature importance image to embed, change path or leave None
EXISTING_FI_IMAGE = None
# e.g. on your machine you had /mnt/data/... - but keep None so script is portable:
# EXISTING_FI_IMAGE = Path(r"C:\path\to\existing_feature_importance.png")

print("PROJECT ROOT:", PROJECT_ROOT)
print("EXPLAIN_DIR:", EXPLAIN_DIR)
print()

# -----------------------
# Load model + data
# -----------------------
print("Loading model and data...")
model_path = MODELS_DIR / "best_pipeline.joblib"
encoder_path = MODELS_DIR / "label_encoder.joblib"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")
if not encoder_path.exists():
    raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data not found at {DATA_PATH}")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)
df = pd.read_csv(DATA_PATH)

TARGET = "CRASH_TYPE"
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in data")

X = df.drop(columns=[TARGET])
y = label_encoder.transform(df[TARGET])

# recreate train/test split to match training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

feature_names = X.columns.tolist()
print(f"Loaded data: X_train={X_train.shape}, X_test={X_test.shape}")

# -----------------------
# SHAP - global (use sample for speed) and local (per-row)
# -----------------------
print("\nComputing SHAP values (global - on a sample)...")
# build explainer (prefer TreeExplainer for tree models)
# note: model may be a pipeline; choose classifier step if available
model_for_explainer = model
try:
    clf = model.named_steps.get("clf", None)
    if clf is not None:
        model_for_explainer = clf
except Exception:
    # model may not be a pipeline
    model_for_explainer = model

use_kernel = False
explainer = None
try:
    explainer = shap.TreeExplainer(model_for_explainer)
except Exception as e:
    print("TreeExplainer not available/failed:", e)
    use_kernel = True

# create a sample for global computation to avoid long runtime
sample_size = min(5000, X_test.shape[0])
X_test_sample = shap.sample(X_test, sample_size, random_state=42)

if use_kernel:
    print("Using KernelExplainer for sampled data (slower).")
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, min(200, len(X_train)), random_state=42))
    shap_values_sample = explainer.shap_values(X_test_sample)
    expected_value = explainer.expected_value
else:
    # TreeExplainer supports DataFrame input and gives faster results.
    explainer = shap.TreeExplainer(model_for_explainer)
    shap_values_sample = explainer.shap_values(X_test_sample)
    expected_value = explainer.expected_value

# compute mean |shap| per feature (handle multiclass list)
if isinstance(shap_values_sample, list):
    # shap_values_sample is list[class] -> arrays (n_samples, n_features)
    abs_mean_per_feature = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values_sample], axis=0)
else:
    abs_mean_per_feature = np.mean(np.abs(shap_values_sample), axis=0)

fi = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': abs_mean_per_feature
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

top20 = fi.head(20).copy()
top20['percent_contribution'] = (top20['mean_abs_shap'] / top20['mean_abs_shap'].sum()) * 100
top20['percent_contribution'] = top20['percent_contribution'].round(2)

top20.to_csv(EXPLAIN_DIR / "shap_top20_percent.csv", index=False)
top20.to_json(EXPLAIN_DIR / "shap_top20_percent.json", orient="records")
print("Saved top20 CSV/JSON.")

# Save a bar plot
plt.figure(figsize=(12,8))
plt.barh(top20['feature'][::-1], top20['mean_abs_shap'][::-1])
plt.xlabel('mean |SHAP value|')
plt.title('Top 20 features by mean absolute SHAP')
plt.tight_layout()
plt.savefig(EXPLAIN_DIR / "shap_top20_bar.png", dpi=200)
plt.close()
print("Saved shap_top20_bar.png")

if EXISTING_FI_IMAGE and Path(EXISTING_FI_IMAGE).exists():
    shutil.copy(EXISTING_FI_IMAGE, EXPLAIN_DIR / "existing_feature_importance.png")

# -----------------------
# Preventive measures (map top features to suggestions)
# -----------------------
preventive_measures = {
    "FIRST_CRASH_TYPE": "Target high-risk collision types with engineering (roundabouts, protected turns), education and enforcement.",
    "LOCATION_CLUSTER": "Install engineering countermeasures (speed cushions, signage), hotspot redesigns and increased enforcement.",
    "CRASH_HOUR": "Focus lighting, policing and awareness campaigns at high-risk hours.",
    "PRIM_CONTRIBUTORY_CAUSE": "Behavioral interventions: distracted driving campaigns, DUI checkpoints, enforcement for speeding.",
    "TRAFFICWAY_TYPE": "Road design changes: medians, separated lanes and speed management.",
    "POSTED_SPEED_LIMIT": "Lower speed limits where needed; speed cameras and traffic calming.",
    "NUM_UNITS": "Improve intersection control and sightlines; redesign merges/diverges.",
    "LIGHTING_CONDITION": "Upgrade street lighting and reflective signage.",
    "WEATHER_CONDITION": "Weather-responsive signage and maintenance in seasonal risk months.",
    "ROADWAY_SURFACE_COND": "Improve pavement maintenance and anti-skid surfacing.",
    "DEVICE_CONDITION": "Maintain traffic control devices and signals.",
    "ROAD_DEFECT": "Promptly repair defects and inspect high-risk sections frequently.",
    "DEFAULT": "Investigate and apply targeted engineering, enforcement or education depending on feature context."
}
top20['preventive_measure'] = top20['feature'].apply(lambda f: preventive_measures.get(f, preventive_measures['DEFAULT']))

# -----------------------
# Local explanations for a few examples: compute per-row (no indexing mismatch)
# -----------------------
print("\nPreparing local explanations (SHAP force + LIME) for example rows...")

# choose positional indices (must be < len(X_test))
example_positions = [10, 100, 250]
example_positions = [p for p in example_positions if p < len(X_test)]
if len(example_positions) == 0:
    example_positions = [0]  # fallback to first row

shap_html_paths = []
lime_html_paths = []

# prepare LIME explainer once
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=list(label_encoder.classes_),
    discretize_continuous=True,
    mode="classification"
)

for pos in example_positions:
    row_df = X_test.iloc[[pos]]  # keep as DataFrame for shap
    row_array = X_test.iloc[pos].values  # 1d array for LIME

    # predict class for this row (use pipeline)
    try:
        pred_proba = model.predict_proba(row_df)[0]
        pred_class_idx = int(np.argmax(pred_proba))
    except Exception:
        # fallback: try model_for_explainer if pipeline mismatch
        pred_proba = model_for_explainer.predict_proba(row_df)[0]
        pred_class_idx = int(np.argmax(pred_proba))

    # SHAP local explanation: compute shap values for this single row using explainer
    try:
        local_shap_vals = explainer.shap_values(row_df)
        # local_shap_vals format: either list of arrays (n_classes) or array
        if isinstance(local_shap_vals, list):
            # choose shap array for predicted class
            shap_for_pred = local_shap_vals[pred_class_idx][0]  # row 0
            expected_val_local = explainer.expected_value[pred_class_idx] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else explainer.expected_value
        else:
            shap_for_pred = local_shap_vals[0] if local_shap_vals.ndim == 2 else local_shap_vals
            expected_val_local = explainer.expected_value
    except Exception as e:
        print(f"Warning: per-row shap_values failed for pos {pos} with error: {e}")
        # as fallback compute shap on a small sample that includes the row
        small_sample = pd.concat([row_df, X_test_sample.head(20)], ignore_index=True)
        local_shap_vals = explainer.shap_values(small_sample)
        if isinstance(local_shap_vals, list):
            shap_for_pred = local_shap_vals[pred_class_idx][0]
            expected_val_local = explainer.expected_value[pred_class_idx] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else explainer.expected_value
        else:
            shap_for_pred = local_shap_vals[0]
            expected_val_local = explainer.expected_value

    # Create SHAP force plot (html)
    try:
        # shap.force_plot wants expected_value and shap values vector
        fp = shap.force_plot(expected_val_local, shap_for_pred, row_df, matplotlib=False)
        html_path = EXPLAIN_DIR / f"shap_force_sample_{pos}.html"
        shap.save_html(str(html_path), fp)
        shap_html_paths.append(html_path.name)
        print(f"Saved SHAP force for sample {pos} -> {html_path.name}")
    except Exception as e:
        print(f"Could not create SHAP force plot for pos {pos}: {e}")

    # LIME explanation
    try:
        lime_exp = lime_explainer.explain_instance(
            data_row=row_array,
            predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names)),
            num_features=10
        )
        lime_path = EXPLAIN_DIR / f"lime_sample_{pos}.html"
        lime_exp.save_to_file(str(lime_path))
        lime_html_paths.append(lime_path.name)
        print(f"Saved LIME explanation for sample {pos} -> {lime_path.name}")
    except Exception as e:
        print(f"Could not create LIME explanation for pos {pos}: {e}")

# -----------------------
# Build central HTML report
# -----------------------
print("\nWriting central HTML report...")

def img_to_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

fi_png = EXPLAIN_DIR / "shap_top20_bar.png"
fi_png_b64 = img_to_base64(fi_png) if fi_png.exists() else ""
existing_fi_b64 = ""
if EXISTING_FI_IMAGE and Path(EXISTING_FI_IMAGE).exists():
    existing_fi_b64 = img_to_base64(Path(EXISTING_FI_IMAGE))

main_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Explainability Report - Road Accident Severity</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 20px; }}
    h1 {{ text-align: center; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    .feature-col {{ width: 30%; }}
    .percent {{ width: 10%; text-align: center; }}
    .measure {{ width: 60%; }}
    .image {{ text-align: center; margin: 20px 0; }}
    .sample-links {{ margin-top: 20px; }}
  </style>
</head>
<body>
  <h1>Explainability Report — Road Accident Severity</h1>

  <h2>Global Feature Importance (Top 20)</h2>
  <div class="image">
    <img src="data:image/png;base64,{fi_png_b64}" alt="SHAP top20" style="max-width:100%; height:auto; border:1px solid #ccc;">
  </div>
"""

if existing_fi_b64:
    main_html += f"""
    <h4>Existing feature importance image (for comparison)</h4>
    <div class="image">
      <img src="data:image/png;base64,{existing_fi_b64}" alt="existing" style="max-width:100%; height:auto; border:1px solid #ccc;">
    </div>
    """

main_html += """
  <h2>Top 20 Features — Percent Contribution & Preventive Measures</h2>
  <table>
    <thead>
      <tr><th class="feature-col">Feature</th><th class="percent">Percent</th><th class="measure">Suggested Preventive Measures</th></tr>
    </thead>
    <tbody>
"""

for _, r in top20.iterrows():
    main_html += f"<tr><td>{r['feature']}</td><td class='percent'>{r['percent_contribution']}%</td><td class='measure'>{r['preventive_measure']}</td></tr>\n"

main_html += """
    </tbody>
  </table>

  <h2>Local Explanations (examples)</h2>
  <p>Below are the saved per-sample SHAP force plots and LIME HTML explanations. Open them to see feature-level contribution for individual records.</p>
  <div class="sample-links">
    <ul>
"""

for name in shap_html_paths:
    main_html += f'<li>SHAP force (sample): <a href="./{name}" target="_blank">{name}</a></li>\n'
for name in lime_html_paths:
    main_html += f'<li>LIME explanation: <a href="./{name}" target="_blank">{name}</a></li>\n'

main_html += """
    </ul>
  </div>

  <hr>
  <p>Report generated by <strong>explainability.py</strong>.</p>
</body>
</html>
"""

main_file = EXPLAIN_DIR / "explainability_report.html"
with open(main_file, "w", encoding="utf-8") as f:
    f.write(main_html)

print("\nReport created at:", main_file)
print("Individual files saved to:", EXPLAIN_DIR)
print("Open explainability_report.html in a browser to view the full report.")
