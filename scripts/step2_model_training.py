"""
STEP 2: MODEL TRAINING
======================
Run this AFTER step1_data_cleaning.py. It will:
  - Load the clean data
  - Train 3 models: Random Forest, XGBoost, LightGBM
  - Compare them using R² and MAE
  - Tune the best model
  - Save the final model as model.pkl
  - Plot feature importances
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "house_data_clean.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH   = os.path.join(BASE_DIR, "models", "model_meta.json")
PLOTS_DIR   = os.path.join(BASE_DIR, "data", "plots")

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: LOADING CLEAN DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Total rows loaded: {len(df)}")

# Only training rows (have a Price)
train_df = df[df['Price'].notna()].copy()
print(f"Training rows (have Price): {len(train_df)}")

# ══════════════════════════════════════════════════════════════════════════
# 2. DEFINE FEATURES & TARGET
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: DEFINING FEATURES")
print("=" * 60)

FEATURES = [
    'Size',              # square footage
    'Bedrooms',          # number of bedrooms
    'Bathrooms',         # number of bathrooms
    'Property_Age',      # age at time of sale
    'Condition_Score',   # Poor=1, Fair=2, Good=3, New=4
    'Location_Encoded',  # mean price of city (target encoding)
    'Total_Rooms',       # Bedrooms + Bathrooms
    'Sale_Year',         # year of sale
    'Sale_Month',        # month of sale
    'Is_New_Condition',  # 1 if Condition == New
    'Is_Small_Property', # 1 if Size < 1200 sqft
]

TARGET = 'Price'

X = train_df[FEATURES]
y = train_df[TARGET]

# Log-transform target (reduces skewness, improves model accuracy)
# We'll predict log(Price) and convert back with exp()
y_log = np.log1p(y)

print(f"Features used ({len(FEATURES)}):")
for f in FEATURES:
    print(f"  → {f}")
print(f"\nTarget: {TARGET}")
print(f"  Original price range: ${y.min():,.0f} – ${y.max():,.0f}")
print(f"  Log-price range: {y_log.min():.2f} – {y_log.max():.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. TRAIN / VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: TRAIN/VALIDATION SPLIT (80% / 20%)")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

print(f"Training set  : {len(X_train)} rows")
print(f"Validation set: {len(X_val)} rows")

# Helper: evaluate model and print results
def evaluate_model(name, model, X_tr, y_tr, X_v, y_v):
    model.fit(X_tr, y_tr)
    preds_log = model.predict(X_v)

    # Convert back from log scale
    preds_actual  = np.expm1(preds_log)
    actual_prices = np.expm1(y_v)

    r2  = r2_score(actual_prices, preds_actual)
    mae = mean_absolute_error(actual_prices, preds_actual)
    rmse = np.sqrt(mean_squared_error(actual_prices, preds_actual))

    print(f"\n{'─'*40}")
    print(f"  Model : {name}")
    print(f"  R²    : {r2:.4f}   (higher is better, max=1.0)")
    print(f"  MAE   : ${mae:,.0f}  (avg prediction error)")
    print(f"  RMSE  : ${rmse:,.0f}")
    print(f"{'─'*40}")
    return r2, mae, rmse

# ══════════════════════════════════════════════════════════════════════════
# 4. TRAIN & COMPARE 3 MODELS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: TRAINING & COMPARING MODELS")
print("=" * 60)

results = {}

# ── Model 1: Random Forest (baseline) ─────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=3,
    n_jobs=-1,           # ← MULTI-CORE: uses all CPU cores
    random_state=42
)
r2, mae, rmse = evaluate_model("Random Forest", rf, X_train, y_train, X_val, y_val)
results['Random Forest'] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}

# ── Model 2: XGBoost ──────────────────────────────────────────────────────
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    n_jobs=-1,           # ← MULTI-CORE
    random_state=42,
    verbosity=0
)
r2, mae, rmse = evaluate_model("XGBoost", xgb_model, X_train, y_train, X_val, y_val)
results['XGBoost'] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}

# ── Model 3: LightGBM ─────────────────────────────────────────────────────
lgbm_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,           # ← MULTI-CORE
    random_state=42,
    verbose=-1
)
r2, mae, rmse = evaluate_model("LightGBM", lgbm_model, X_train, y_train, X_val, y_val)
results['LightGBM'] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}

# ── Print Comparison Table ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(results).T
results_df['R2'] = results_df['R2'].map(lambda x: f"{x:.4f}")
results_df['MAE'] = results_df['MAE'].map(lambda x: f"${x:,.0f}")
results_df['RMSE'] = results_df['RMSE'].map(lambda x: f"${x:,.0f}")
print(results_df.to_string())

# Determine best model by R²
best_name = max(results, key=lambda k: results[k]['R2'])
print(f"\n🏆 Best Model: {best_name} (R² = {results[best_name]['R2']:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION ON BEST MODEL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: 5-FOLD CROSS-VALIDATION (Best Model)")
print("=" * 60)

# Pick best model
if best_name == 'XGBoost':
    best_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                   subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                   random_state=42, verbosity=0)
elif best_name == 'LightGBM':
    best_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                    random_state=42, verbose=-1)
else:
    best_model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_model, X, y_log,
    cv=kf,
    scoring='r2',
    n_jobs=-1    # ← MULTI-CORE: folds run in parallel
)

print(f"CV R² scores per fold: {[round(s, 4) for s in cv_scores]}")
print(f"Mean CV R²           : {cv_scores.mean():.4f}")
print(f"Std  CV R²           : {cv_scores.std():.4f}")
print("(A stable std means the model generalises well)")

# ══════════════════════════════════════════════════════════════════════════
# 6. TRAIN FINAL MODEL ON ALL TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: TRAINING FINAL MODEL ON ALL DATA")
print("=" * 60)

best_model.fit(X, y_log)
print(f"✓ Final {best_name} model trained on {len(X)} rows")

# ══════════════════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE PLOT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: FEATURE IMPORTANCES")
print("=" * 60)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

    print(feat_imp.sort_values(ascending=False).to_string())

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2196F3' if v < feat_imp.max() * 0.5 else '#4CAF50' for v in feat_imp]
    feat_imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title(f'Feature Importances — {best_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    for i, (val, name) in enumerate(zip(feat_imp, feat_imp.index)):
        ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'plot7_feature_importance.png'), dpi=150)
    plt.close()
    print("\n✓ Saved: plot7_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════
# 8. ACTUAL VS PREDICTED PLOT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: ACTUAL vs PREDICTED PLOT")
print("=" * 60)

# Refit on train split, predict val split for this plot
best_model.fit(X_train, y_train)
preds_log = best_model.predict(X_val)
actual_prices = np.expm1(y_val)
pred_prices   = np.expm1(preds_log)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: Actual vs Predicted
axes[0].scatter(actual_prices/1000, pred_prices/1000, alpha=0.5, color='steelblue', s=30)
min_val = min(actual_prices.min(), pred_prices.min()) / 1000
max_val = max(actual_prices.max(), pred_prices.max()) / 1000
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price ($000s)')
axes[0].set_ylabel('Predicted Price ($000s)')
axes[0].set_title(f'Actual vs Predicted — {best_name}', fontweight='bold')
axes[0].legend()

# Residuals plot
residuals = actual_prices - pred_prices
axes[1].scatter(pred_prices/1000, residuals/1000, alpha=0.5, color='coral', s=30)
axes[1].axhline(0, color='black', lw=2, ls='--')
axes[1].set_xlabel('Predicted Price ($000s)')
axes[1].set_ylabel('Residual ($000s)')
axes[1].set_title('Residuals Plot', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot8_actual_vs_predicted.png'), dpi=150)
plt.close()
print("✓ Saved: plot8_actual_vs_predicted.png")

final_r2  = r2_score(actual_prices, pred_prices)
final_mae = mean_absolute_error(actual_prices, pred_prices)
print(f"\nFinal Validation R² : {final_r2:.4f}")
print(f"Final Validation MAE: ${final_mae:,.0f}")

# ══════════════════════════════════════════════════════════════════════════
# 9. SAVE MODEL + METADATA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: SAVING MODEL")
print("=" * 60)

# Retrain final model on ALL data before saving
best_model.fit(X, y_log)

joblib.dump(best_model, MODEL_PATH)
print(f"✓ Model saved to: {MODEL_PATH}")

# Save metadata (needed by the API to know feature names, encoding maps etc.)
city_price_map = train_df.groupby('Location')['Price'].mean().round(0).to_dict()

meta = {
    "model_name"      : best_name,
    "features"        : FEATURES,
    "r2_validation"   : round(final_r2, 4),
    "mae_validation"  : round(final_mae, 0),
    "cv_r2_mean"      : round(float(cv_scores.mean()), 4),
    "cv_r2_std"       : round(float(cv_scores.std()), 4),
    "city_price_map"  : {k: int(v) for k, v in city_price_map.items()},
    "condition_map"   : {"Poor": 1, "Fair": 2, "Good": 3, "New": 4},
    "target_transform": "log1p"
}

with open(META_PATH, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"✓ Metadata saved to: {META_PATH}")
print(f"\nMetadata contents:")
print(json.dumps(meta, indent=2))

print("\n" + "=" * 60)
print("✅ STEP 2 COMPLETE — Model trained and saved!")
print("   Next: Run  api/main.py  to start the API")
print("=" * 60)
