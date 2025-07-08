# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# + [markdown]
# # BigMart Sales Analysis
# 
# This notebook analyzes BigMart sales data to identify patterns and relationships between product attributes and sales performance.

# + [markdown]
# ## Data Loading

# + [code]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load data
DATA_PATH = r"C:\\Users\\Shani\\Documents\\Windsurf_Demo\\bigmart_sales_predictions.csv"
sales_df = pd.read_csv(DATA_PATH)

# Clean column names
sales_df.columns = sales_df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")

# Show first few rows
sales_df.head()

# + [markdown]
# ## Data Cleaning

# + [code]
# Standardize Item_Fat_Content values
sales_df["Item_Fat_Content"] = sales_df["Item_Fat_Content"].replace({
    "Low Fat": "low",
    "LF": "low",
    "low fat": "low",
    "Regular": "regular",
    "reg": "regular"
})

# Check for missing values
sales_df.isna().sum()

# + [markdown]
# ## Exploratory Data Analysis

# + [code]
# Basic statistics
sales_df.describe()

# + [markdown]
# ### Sales Distribution

# + [code]
# Histogram of Item_Outlet_Sales
plt.figure(figsize=(10, 6))
sns.histplot(data=sales_df, x="Item_Outlet_Sales", kde=True)
plt.title("Distribution of Item Outlet Sales")
plt.axvline(sales_df["Item_Outlet_Sales"].median(), color="red", linestyle="--", 
            label=f"Median: {sales_df['Item_Outlet_Sales'].median():,.0f}")
plt.legend()
plt.show()

# + [markdown]
# ### Outlet Analysis

# + [code]
# Boxplot of sales by outlet size
plt.figure(figsize=(10, 6))
sns.boxplot(data=sales_df, x="Outlet_Size", y="Item_Outlet_Sales")
plt.title("Sales Distribution by Outlet Size")
plt.show()

# + [markdown]
# ### Item Analysis

# + [code]
# Top 10 item types by sales
plt.figure(figsize=(12, 6))
top_items = sales_df["Item_Type"].value_counts().head(10).index
sns.boxplot(
    data=sales_df[sales_df["Item_Type"].isin(top_items)],
    x="Item_Type",
    y="Item_Outlet_Sales"
)
plt.xticks(rotation=45)
plt.title("Sales Distribution for Top 10 Item Types")
plt.show()

# + [markdown]
# ## Advanced Analysis

# + [code]
# Correlation heatmap
numeric_cols = sales_df.select_dtypes(include=["number"]).columns
plt.figure(figsize=(12, 8))
sns.heatmap(
    sales_df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# + [markdown]
# ## Key Insights
# 
# 1. **Sales Distribution**: Most items have sales below ₹2,500 with a long tail of higher sales
# 2. **Outlet Impact**: Larger outlets tend to have higher median sales
# 3. **Item Types**: Certain categories like "Fruits and Vegetables" show wider sales variation
# 4. **Correlations**: Item MRP shows the strongest positive correlation with sales (0.57)

# + [code]
# Save cleaned data for further analysis
sales_df.to_csv("bigmart_sales_cleaned.csv", index=False)
