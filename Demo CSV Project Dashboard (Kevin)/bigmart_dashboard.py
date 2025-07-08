import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------
# 1. Page configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="BigMart Sales Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 2. Utility functions
# -------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the BigMart sales dataset and perform basic cleaning.

    The function is cached so it will only run once per session unless the
    underlying file changes.
    """
    @st.cache_data(hash_funcs={Path: lambda _: None})
    def _load(path_str: str) -> pd.DataFrame:
        df = pd.read_csv(path_str)
        # Standardise column names (lower snake_case)
        df.columns = (
            df.columns.str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("/", "_", regex=False)
        )

        # Replace inconsistent fat content labels, mirroring the template_EDA
        if "Item_Fat_Content" in df.columns:
            df["Item_Fat_Content"] = (
                df["Item_Fat_Content"].replace(
                    {
                        "Low Fat": "low",
                        "LF": "low",
                        "low fat": "low",
                        "Regular": "regular",
                        "reg": "regular",
                    }
                )
            )

        return df

    return _load(str(csv_path))


def numeric_columns(df: pd.DataFrame):
    """Return columns with numeric dtype."""
    return df.select_dtypes(include="number").columns.tolist()


# -------------------------------------------------------------
# 3. Sidebar – data selection & filters
# -------------------------------------------------------------

DATA_PATH = Path(r"C:\Users\Shani\Documents\Windsurf_Demo\bigmart_synthetic_data.csv")

# Load data once and reuse
try:
    sales_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found at {DATA_PATH}. Please verify the path.")
    st.stop()

st.sidebar.header("Filters")

# Dynamic filters based on columns existing in the dataset
item_types = sorted(sales_df["Item_Type"].unique()) if "Item_Type" in sales_df.columns else []
outlet_ids = sorted(sales_df["Outlet_Identifier"].unique()) if "Outlet_Identifier" in sales_df.columns else []

selected_item_types = st.sidebar.multiselect(
    "Select Item Types", options=item_types, default=item_types if item_types else None
)

selected_outlets = st.sidebar.multiselect(
    "Select Outlet IDs", options=outlet_ids, default=outlet_ids if outlet_ids else None
)

# Apply filters
filtered_df = sales_df.copy()
if selected_item_types:
    filtered_df = filtered_df[filtered_df["Item_Type"].isin(selected_item_types)]
if selected_outlets:
    filtered_df = filtered_df[filtered_df["Outlet_Identifier"].isin(selected_outlets)]

# -------------------------------------------------------------
# 4. Key Metrics
# -------------------------------------------------------------

st.title("🛒 BigMart Sales Analysis Dashboard")

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    total_sales = filtered_df.get("Item_Outlet_Sales", pd.Series(dtype=float)).sum()
    st.metric("Total Sales", f"₹{total_sales:,.0f}")

with col2:
    num_items = filtered_df.shape[0]
    st.metric("Total Records", f"{num_items:,}")

with col3:
    avg_mrp = filtered_df.get("Item_MRP", pd.Series(dtype=float)).mean()
    st.metric("Average MRP", f"₹{avg_mrp:,.2f}")

# -------------------------------------------------------------
# 5. Visualisations – First Iteration
# -------------------------------------------------------------

st.header("Exploratory Visualisations")

# Using two columns layout for charts
hist_col, box_col = st.columns(2)

# Histogram – Item Outlet Sales
if "Item_Outlet_Sales" in filtered_df.columns:
    with hist_col:
        fig, ax = plt.subplots(figsize=(6, 4))
        median_sales = filtered_df["Item_Outlet_Sales"].median()
        sns.histplot(data=filtered_df, x="Item_Outlet_Sales", kde=False, ax=ax)
        ax.axvline(median_sales, color="black", linestyle="--", label=f"Median = {median_sales:,.0f}")
        ax.set_title("Distribution of Item Outlet Sales")
        ax.legend()
        st.pyplot(fig)

# Boxplot – Outlet Size vs Sales
if {"Outlet_Size", "Item_Outlet_Sales"}.issubset(filtered_df.columns):
    with box_col:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=filtered_df, x="Outlet_Size", y="Item_Outlet_Sales", ax=ax)
        ax.set_title("Sales by Outlet Size")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

# -------------------------------------------------------------
# 6. Additional Visualisations – Template Reproductions
# -------------------------------------------------------------

add_tab1, add_tab2, add_tab3, add_tab4, add_tab5 = st.tabs(
    ["Item MRP Distribution",
     "Outlet Location vs Sales",
     "Item Type Violinplot",
     "Correlation Heatmap",
     "MRP vs Sales Scatter"]
)

# Histogram – Item_MRP
with add_tab1:
    if "Item_MRP" in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        median_mrp = filtered_df["Item_MRP"].median()
        sns.histplot(data=filtered_df, x="Item_MRP", kde=False, ax=ax)
        ax.axvline(median_mrp, color="black", linestyle="--", label=f"Median = {median_mrp:,.0f}")
        ax.set_title("Distribution of Item MRP")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Item_MRP column not in dataset.")

# Boxplot – Outlet_Location_Type vs Sales
with add_tab2:
    if {"Outlet_Location_Type", "Item_Outlet_Sales"}.issubset(filtered_df.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=filtered_df, x="Outlet_Location_Type", y="Item_Outlet_Sales", ax=ax)
        ax.set_title("Sales by Outlet Location Type")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    else:
        st.info("Required columns not present.")

# Violinplot – Item_Type vs Sales
with add_tab3:
    if {"Item_Type", "Item_Outlet_Sales"}.issubset(filtered_df.columns):
        top_types = filtered_df["Item_Type"].value_counts().head(10).index
        df_violin = filtered_df[filtered_df["Item_Type"].isin(top_types)]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=df_violin, x="Item_Type", y="Item_Outlet_Sales", ax=ax)
        ax.set_title("Sales Distribution by Item Type (Top 10)")
        ax.tick_params(axis="x", rotation=90)
        st.pyplot(fig)
    else:
        st.info("Required columns not present.")

# Correlation Heatmap of numeric features
with add_tab4:
    num_cols = numeric_columns(filtered_df)
    if len(num_cols) > 1:
        corr = filtered_df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, cmap="Greens", annot=True, ax=ax)
        ax.set_title("Correlation Heatmap – Numeric Features")
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# Scatterplot – Item_MRP vs Sales coloured by Outlet_Type
with add_tab5:
    if {"Item_MRP", "Item_Outlet_Sales", "Outlet_Type"}.issubset(filtered_df.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=filtered_df, x="Item_MRP", y="Item_Outlet_Sales", hue="Outlet_Type", palette="plasma", ax=ax)
        ax.set_title("Item MRP vs Item Outlet Sales by Outlet Type")
        st.pyplot(fig)
    else:
        st.info("Required columns not present.")
