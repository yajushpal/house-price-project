"""
STEP 1: DATA CLEANING & EXPLORATORY DATA ANALYSIS (EDA)
=========================================================
Run this file first. It will:
  - Load the raw Excel data
  - Fix the Date column
  - Handle missing values
  - Engineer new features
  - Save a clean CSV ready for modelling
  - Print a full EDA summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "house_data.xlsx")
OUT_PATH   = os.path.join(BASE_DIR, "data", "house_data_clean.csv")
PLOTS_DIR  = os.path.join(BASE_DIR, "data", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"Raw shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# ══════════════════════════════════════════════════════════════════════════
# 2. INITIAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: DATA OVERVIEW")
print("=" * 60)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values (before cleaning):")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

print(f"\nRows with NO Price (inference rows): {df['Price'].isna().sum()}")
print(f"Rows WITH Price    (training rows):  {df['Price'].notna().sum()}")

# ══════════════════════════════════════════════════════════════════════════
# 3. FIX DATE COLUMN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: FIXING DATE COLUMN")
print("=" * 60)

# Excel stores dates as integer serial numbers (days since 1899-12-30)
# e.g. 44137 → 2020-11-01
df["Date_Sold"] = pd.to_datetime(df["Date Sold"])
df['Sale_Year']  = df['Date_Sold'].dt.year
df['Sale_Month'] = df['Date_Sold'].dt.month

print(f"Sample Date Sold values (raw):    {df['Date Sold'].head(3).tolist()}")
print(f"Sample Date Sold values (fixed):  {df['Date_Sold'].head(3).tolist()}")
print(f"Sale Year range: {df['Sale_Year'].min()} – {df['Sale_Year'].max()}")
print(f"Sale Month range: {df['Sale_Month'].min()} – {df['Sale_Month'].max()}")

# ══════════════════════════════════════════════════════════════════════════
# 4. SEPARATE TRAIN vs INFERENCE ROWS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: SEPARATING TRAIN & INFERENCE ROWS")
print("=" * 60)

# Rows without Price cannot be used for training — save them separately
train_df = df[df['Price'].notna()].copy()
infer_df = df[df['Price'].isna()].copy()

print(f"Training rows   (have Price):  {len(train_df)}")
print(f"Inference rows  (no Price):    {len(infer_df)}")
print(f"Inference Property IDs: {infer_df['Property ID'].tolist()}")

# ══════════════════════════════════════════════════════════════════════════
# 5. ENCODE CONDITION (Ordinal — it has a natural order)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: ENCODING CONDITION")
print("=" * 60)

condition_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'New': 4}
df['Condition_Score'] = df['Condition'].map(condition_map)

print("Condition → Score mapping:", condition_map)
print(df[['Condition', 'Condition_Score']].drop_duplicates().dropna().sort_values('Condition_Score'))

# ══════════════════════════════════════════════════════════════════════════
# 6. LOCATION ENCODING (Target Encoding — mean price per city)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: LOCATION ENCODING (Target Encoding)")
print("=" * 60)

# IMPORTANT: compute mean price only from TRAINING rows to avoid data leakage
city_price_map = train_df.groupby('Location')['Price'].mean().round(0)
print("Mean Price per City (used for encoding):")
print(city_price_map)

df['Location_Encoded'] = df['Location'].map(city_price_map)

# ══════════════════════════════════════════════════════════════════════════
# 7. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: FEATURE ENGINEERING")
print("=" * 60)

# Age of property at time of sale
df['Property_Age'] = df['Sale_Year'] - df['Year Built']
# Clip negative values (data errors where year built > sale year)
df['Property_Age'] = df['Property_Age'].clip(lower=0)

# Total rooms
df['Total_Rooms'] = df['Bedrooms'].fillna(0) + df['Bathrooms'].fillna(0)

# Is it a new/recently built property?
df['Is_New_Condition'] = (df['Condition'] == 'New').astype(float)

# Small property flag (< 1200 sqft)
df['Is_Small_Property'] = (df['Size'] < 1200).astype(float)

print("New features created:")
print("  → Property_Age    : Sale Year − Year Built")
print("  → Total_Rooms     : Bedrooms + Bathrooms")
print("  → Is_New_Condition: 1 if Condition == New")
print("  → Is_Small_Property: 1 if Size < 1200 sqft")

# ══════════════════════════════════════════════════════════════════════════
# 8. HANDLE MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: IMPUTING MISSING VALUES")
print("=" * 60)

# Strategy: use median grouped by Location (CityA houses differ from CityD)
# This is smarter than a global median

def impute_by_location(df, col):
    """Fill missing values with median of same city."""
    median_by_loc = df.groupby('Location')[col].transform('median')
    df[col] = df[col].fillna(median_by_loc)
    # If still missing (whole city is NaN), use global median
    df[col] = df[col].fillna(df[col].median())
    return df

for col in ['Size', 'Bedrooms', 'Bathrooms']:
    before = df[col].isna().sum()
    df = impute_by_location(df, col)
    after = df[col].isna().sum()
    print(f"  {col}: {before} missing → {after} missing (filled with city median)")

# Year Built — impute with city median
before = df['Year Built'].isna().sum()
df = impute_by_location(df, 'Year Built')
after = df['Year Built'].isna().sum()
print(f"  Year Built: {before} missing → {after} missing (filled with city median)")

# Recompute Property_Age after Year Built imputation
df['Property_Age'] = (df['Sale_Year'] - df['Year Built']).clip(lower=0)

# Condition_Score — impute with city mode
before = df['Condition_Score'].isna().sum()
mode_by_loc = df.groupby('Location')['Condition_Score'].transform(lambda x: x.mode()[0] if not x.mode().empty else 2)
df['Condition_Score'] = df['Condition_Score'].fillna(mode_by_loc)
after = df['Condition_Score'].isna().sum()
print(f"  Condition: {before} missing → {after} missing (filled with city mode)")

# Location_Encoded — should have no missing now
df['Location_Encoded'] = df['Location_Encoded'].fillna(df['Location_Encoded'].median())

print("\nMissing values AFTER cleaning:")
print(df[['Size','Bedrooms','Bathrooms','Year Built','Condition_Score','Property_Age']].isnull().sum())

# ══════════════════════════════════════════════════════════════════════════
# 9. EDA — PRICE ANALYSIS (only on training rows)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: EDA — PRICE ANALYSIS")
print("=" * 60)

train_clean = df[df['Price'].notna()].copy()

print(f"\nPrice Statistics:")
print(train_clean['Price'].describe().apply(lambda x: f"${x:,.0f}"))

print(f"\nPrice by City:")
print(train_clean.groupby('Location')['Price'].agg(['mean','median','min','max']).map(lambda x: f"${x:,.0f}"))

print(f"\nPrice by Condition:")
print(train_clean.groupby('Condition')['Price'].agg(['mean','median']).map(lambda x: f"${x:,.0f}"))

# ── Plot 1: Price Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_clean['Price'] / 1000, bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Price (in $000s)')
axes[0].set_ylabel('Count')

axes[1].hist(np.log1p(train_clean['Price']), bins=40, color='seagreen', edgecolor='white')
axes[1].set_title('Log(Price) Distribution — More Normal', fontsize=14, fontweight='bold')
axes[1].set_xlabel('log(Price)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot1_price_distribution.png'), dpi=150)
plt.close()
print("\n✓ Saved: plot1_price_distribution.png")

# ── Plot 2: Price by City ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
city_order = train_clean.groupby('Location')['Price'].median().sort_values(ascending=False).index
sns.boxplot(data=train_clean, x='Location', y='Price', order=city_order,
            palette='Set2', ax=ax)
ax.set_title('Price Distribution by City', fontsize=14, fontweight='bold')
ax.set_xlabel('City')
ax.set_ylabel('Price ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot2_price_by_city.png'), dpi=150)
plt.close()
print("✓ Saved: plot2_price_by_city.png")

# ── Plot 3: Price by Condition ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
cond_order = ['Poor', 'Fair', 'Good', 'New']
sns.boxplot(data=train_clean, x='Condition', y='Price', order=cond_order,
            palette='RdYlGn', ax=ax)
ax.set_title('Price Distribution by Condition', fontsize=14, fontweight='bold')
ax.set_xlabel('Condition')
ax.set_ylabel('Price ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot3_price_by_condition.png'), dpi=150)
plt.close()
print("✓ Saved: plot3_price_by_condition.png")

# ── Plot 4: Price vs Size ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(train_clean['Size'], train_clean['Price'] / 1000,
                     c=train_clean['Condition_Score'], cmap='RdYlGn',
                     alpha=0.6, s=40)
plt.colorbar(scatter, ax=ax, label='Condition Score (1=Poor, 4=New)')
ax.set_title('Price vs Size (coloured by Condition)', fontsize=14, fontweight='bold')
ax.set_xlabel('Size (sq ft)')
ax.set_ylabel('Price ($000s)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot4_price_vs_size.png'), dpi=150)
plt.close()
print("✓ Saved: plot4_price_vs_size.png")

# ── Plot 5: Correlation Heatmap ───────────────────────────────────────────
num_cols = ['Price', 'Size', 'Bedrooms', 'Bathrooms', 'Property_Age',
            'Condition_Score', 'Location_Encoded', 'Total_Rooms']
corr = train_clean[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot5_correlation_heatmap.png'), dpi=150)
plt.close()
print("✓ Saved: plot5_correlation_heatmap.png")

# ── Plot 6: Missing Value Heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
miss_data = df[['Size','Bedrooms','Bathrooms','Year Built','Condition','Price']].isnull()
sns.heatmap(miss_data.T, cbar=False, yticklabels=True, cmap='Reds', ax=ax)
ax.set_title('Missing Values Heatmap (Red = Missing)', fontsize=13, fontweight='bold')
ax.set_xlabel('Row Index')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot6_missing_values.png'), dpi=150)
plt.close()
print("✓ Saved: plot6_missing_values.png")

# ══════════════════════════════════════════════════════════════════════════
# 10. SAVE CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10: SAVING CLEAN DATA")
print("=" * 60)

# Final feature set
keep_cols = [
    'Property ID', 'Location', 'Size', 'Bedrooms', 'Bathrooms',
    'Year Built', 'Condition', 'Date_Sold', 'Sale_Year', 'Sale_Month',
    'Price', 'Condition_Score', 'Location_Encoded',
    'Property_Age', 'Total_Rooms', 'Is_New_Condition', 'Is_Small_Property'
]

df_clean = df[keep_cols].copy()
df_clean.to_csv(OUT_PATH, index=False)

print(f"✓ Clean data saved to: {OUT_PATH}")
print(f"  Shape: {df_clean.shape}")
print(f"\nFinal columns:")
for c in df_clean.columns:
    print(f"  → {c}")

print("\n" + "=" * 60)
print("✅ STEP 1 COMPLETE — Data is clean and ready for modelling!")
print("   Next: Run  scripts/step2_model_training.py")
print("=" * 60)
