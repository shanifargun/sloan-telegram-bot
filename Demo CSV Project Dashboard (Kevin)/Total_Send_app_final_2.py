import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import datetime

# Set page config
st.set_page_config(
    page_title="Medical Test Send-out Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dashboard
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric:hover {
        background-color: #e0e2e6;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading
@st.cache_data
def load_data():
    df = pd.read_csv('https://www.dropbox.com/scl/fi/98v40l05hk8p7ctkdozx7/Total_Send_JanSep24.csv?rlkey=kc5jfz1cm65239wacv4254q47&st=ll8dtkx6&dl=1', encoding='latin1')
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('\n', '', regex=True)
    
    # Clean and convert Price and Total_Cost columns
    for col in ['Price', 'Total_Cost']:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('', np.nan)
        df[col] = pd.to_numeric(df[col].str.replace('$', '').str.replace(',', ''), errors='coerce')
    
    # Ensure Volume is numeric
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Drop rows where all key columns are NaN
    df.dropna(subset=['Volume', 'Price', 'Total_Cost'], how='all', inplace=True)
    
    # Clean Department and Lab columns
    df['Department'] = df['Department'].fillna('Unknown')
    df['Lab'] = df['Lab'].fillna('Unknown')
    
    return df

# Title and description
st.title("Medical Test Send-out Analysis Dashboard")

# Dashboard Summary Card
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h3>Dashboard Overview</h3>
        <p>This interactive dashboard analyzes medical test send-out data to identify:</p>
        <ul>
            <li>Cost and volume patterns across departments and labs</li>
            <li>ABC analysis for test prioritization</li>
            <li>Key opportunities for cost optimization</li>
            <li>Test validation tracking and cost savings calculation</li>
        </ul>
        <p><i>Click on charts to explore detailed department and lab analysis. Use the department selection to access the test validation features.</i></p>
    </div>
""", unsafe_allow_html=True)

# Load the data
try:
    df = load_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Security warning
st.warning("""⚠️ SECURITY NOTICE: This dashboard contains sensitive medical data. 
    - Only run this locally on your machine
    - Do not share or deploy this dashboard publicly
    - Ensure you're in a private setting when viewing the data""")

# Move filters here, after data is confirmed to be loaded
st.sidebar.header("Filters")
    
# Get unique departments and labs, handling missing values
dept_options = sorted(df['Department'].unique().tolist())
lab_options = sorted(df['Lab'].unique().tolist())
    
# Default to all options selected
selected_departments = st.sidebar.multiselect(
    "Select Departments",
    options=dept_options,
    default=dept_options
)

selected_labs = st.sidebar.multiselect(
    "Select Labs",
    options=lab_options,
    default=lab_options
)
    
# Filter the dataframe
filtered_df = df[
    (df['Department'].isin(selected_departments)) &
    (df['Lab'].isin(selected_labs))
]

# Key Metrics Section
st.header("Key Metrics & Insights")

# Create three columns for metrics
col1, col2, col3 = st.columns(3)

# Calculate metrics
total_cost = filtered_df['Total_Cost'].sum()
total_volume = filtered_df['Volume'].sum()
avg_cost_per_test = total_cost / total_volume if total_volume > 0 else 0
unique_tests = filtered_df['Test_Name'].nunique()
total_labs = filtered_df['Lab'].nunique()

# Display metrics with formatted values
with col1:
    st.metric(
        "Total Cost",
        f"${total_cost:,.2f}",
        delta=None,
        help="Total cost of all selected send-out tests"
    )
    st.metric(
        "Unique Tests",
        f"{unique_tests:,}",
        delta=None,
        help="Number of unique test types"
    )

with col2:
    st.metric(
        "Total Volume",
        f"{total_volume:,.0f}",
        delta=None,
        help="Total volume of tests performed"
    )
    st.metric(
        "Number of Labs",
        f"{total_labs:,}",
        delta=None,
        help="Number of unique reference laboratories"
    )

with col3:
    st.metric(
        "Average Cost per Test",
        f"${avg_cost_per_test:,.2f}",
        delta=None,
        help="Average cost per test across all selected departments"
    )

# Department Analysis Section
st.header("Department Analysis")

# Create department summary
dept_summary = filtered_df.groupby('Department').agg({
    'Total_Cost': 'sum',
    'Volume': 'sum',
    'Test_Name': 'nunique'
}).reset_index()

dept_summary['Cost_per_Test'] = dept_summary['Total_Cost'] / dept_summary['Volume']
dept_summary = dept_summary.sort_values('Total_Cost', ascending=True)

# Create interactive department bar chart
fig = go.Figure()

# Add Total Cost bars
fig.add_trace(
    go.Bar(
        x=dept_summary['Total_Cost'],
        y=dept_summary['Department'],
        orientation='h',
        name='Total Cost',
        text=dept_summary['Total_Cost'].apply(lambda x: f'${x:,.2f}'),
        hovertemplate="<b>%{y}</b><br>" +
                     "Total Cost: %{text}<br>" +
                     "<extra></extra>",
        marker_color='#1f77b4'
    )
)

# Update layout
fig.update_layout(
    title='Department Cost Analysis',
    height=400,
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis_title='Total Cost ($)',
    yaxis_title='Department',
    showlegend=False,
    hovermode='closest'
)

# Add click event instruction
st.markdown("*Click on a department bar to view detailed analysis*")

# Display the plot
selected_dept = st.plotly_chart(fig, use_container_width=True)

# Cost Analysis Section with Tabs
st.header("Cost Analysis")
cost_tab1, cost_tab2 = st.tabs(["Department Analysis", "Lab Analysis"])

with cost_tab1:
    # Department Analysis Tab
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost Distribution by Department - Handle Outliers
        # Calculate outlier thresholds using IQR method
        Q1 = filtered_df['Total_Cost'].quantile(0.25)
        Q3 = filtered_df['Total_Cost'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR
        
        # Create filtered dataset for boxplot
        boxplot_df = filtered_df[filtered_df['Total_Cost'] <= outlier_threshold].copy()
        
        # Create boxplot without outliers
        dept_dist = px.box(
            boxplot_df,
            x='Department',
            y='Total_Cost',
            title='Cost Distribution by Department (Excluding Extreme Outliers)',
            labels={'Total_Cost': 'Total Cost ($)', 'Department': 'Department'},
            height=400
        )
        dept_dist.update_traces(quartilemethod="linear")
        dept_dist.update_layout(
            showlegend=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(dept_dist, use_container_width=True)
        
        # Show outlier information
        outliers = filtered_df[filtered_df['Total_Cost'] > outlier_threshold]
        if not outliers.empty:
            with st.expander("View Outlier Tests"):
                st.write("Tests excluded from boxplot due to extreme values:")
                outlier_summary = outliers[['Test_Name', 'Department', 'Total_Cost']].sort_values('Total_Cost', ascending=False)
                st.dataframe(
                    outlier_summary.style.format({
                        'Total_Cost': '${:,.2f}'
                    })
                )
    
    with col2:
        # Volume vs Cost Scatter
        dept_scatter = px.scatter(
            dept_summary,
            x='Volume',
            y='Total_Cost',
            size='Test_Name',
            color='Department',
            hover_name='Department',
            title='Volume vs Cost by Department',
            labels={
                'Volume': 'Total Volume',
                'Total_Cost': 'Total Cost ($)',
                'Test_Name': 'Number of Unique Tests'
            },
            height=400
        )
        st.plotly_chart(dept_scatter, use_container_width=True)

with cost_tab2:
    # Lab Analysis Tab
    # Create lab summary
    lab_summary = filtered_df.groupby('Lab').agg({
        'Total_Cost': 'sum',
        'Volume': 'sum',
        'Test_Name': 'nunique',
        'Department': 'nunique'
    }).reset_index()
    
    lab_summary['Cost_per_Test'] = lab_summary['Total_Cost'] / lab_summary['Volume']
    lab_summary = lab_summary.sort_values('Total_Cost', ascending=False)
    
    # Top 10 Labs Analysis
    st.subheader("Top 10 Labs Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 Labs by Cost
        top_labs_cost = px.bar(
            lab_summary.head(10),
            x='Lab',
            y='Total_Cost',
            title='Top 10 Labs by Total Cost',
            labels={'Total_Cost': 'Total Cost ($)', 'Lab': 'Laboratory'},
            height=400
        )
        top_labs_cost.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(top_labs_cost, use_container_width=True)
    
    with col2:
        # Lab Cost vs Volume
        lab_scatter = px.scatter(
            lab_summary,
            x='Volume',
            y='Total_Cost',
            size='Test_Name',
            color='Department',
            hover_name='Lab',
            title='Lab Volume vs Cost Analysis',
            labels={
                'Volume': 'Total Volume',
                'Total_Cost': 'Total Cost ($)',
                'Test_Name': 'Number of Unique Tests'
            },
            height=400
        )
        st.plotly_chart(lab_scatter, use_container_width=True)

    # Lab Details Table
    st.subheader("Lab Details")
    lab_details = lab_summary.copy()
    lab_details['Total_Cost'] = lab_details['Total_Cost'].apply(lambda x: f"${x:,.2f}")
    lab_details['Cost_per_Test'] = lab_details['Cost_per_Test'].apply(lambda x: f"${x:,.2f}")
    lab_details.columns = ['Lab', 'Total Cost', 'Volume', 'Unique Tests', 
                          'Departments Served', 'Cost per Test']
    
    st.dataframe(
        lab_details,
        column_config={
            "Lab": st.column_config.TextColumn("Lab", width="medium"),
            "Total Cost": st.column_config.TextColumn("Total Cost", width="medium"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            "Unique Tests": st.column_config.NumberColumn("Unique Tests", format="%d"),
            "Departments Served": st.column_config.NumberColumn("Departments", format="%d"),
            "Cost per Test": st.column_config.TextColumn("Cost per Test", width="medium"),
        },
        hide_index=True,
    )

# Advanced Analytics Section
st.header("Advanced Analytics")

# Create tabs for different analyses
analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
    "Cost Efficiency Analysis", 
    "Volume-Cost Analysis",
    "Vendor Analysis"
])

with analysis_tab1:
    st.subheader("Cost Efficiency Analysis")
    
    # Calculate efficiency metrics
    dept_efficiency = filtered_df.groupby('Department').agg({
        'Total_Cost': 'sum',
        'Volume': 'sum',
        'Test_Name': 'nunique',
        'Lab': 'nunique'
    }).reset_index()
    
    dept_efficiency['Cost_per_Test'] = dept_efficiency['Total_Cost'] / dept_efficiency['Volume']
    dept_efficiency['Tests_per_Lab'] = dept_efficiency['Test_Name'] / dept_efficiency['Lab']
    dept_efficiency = dept_efficiency.sort_values('Cost_per_Test', ascending=True)

    col1, col2 = st.columns(2)
    
    with col1:
        # Cost per Test by Department
        fig = px.bar(
            dept_efficiency,
            x='Department',
            y='Cost_per_Test',
            title='Average Cost per Test by Department',
            labels={'Cost_per_Test': 'Cost per Test ($)', 'Department': 'Department'},
            text=dept_efficiency['Cost_per_Test'].apply(lambda x: f'${x:,.2f}')
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Tests per Lab Ratio
        fig = px.bar(
            dept_efficiency,
            x='Department',
            y='Tests_per_Lab',
            title='Test Concentration (Tests per Lab)',
            labels={'Tests_per_Lab': 'Tests per Lab', 'Department': 'Department'},
            text=dept_efficiency['Tests_per_Lab'].apply(lambda x: f'{x:.1f}')
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

with analysis_tab2:
    st.subheader("Volume-Cost Relationship Analysis")
    
    # Calculate volume-cost metrics
    test_metrics = filtered_df.groupby('Test_Name').agg({
        'Total_Cost': 'sum',
        'Volume': 'sum',
        'Department': 'first',
        'Lab': 'first'
    }).reset_index()
    
    test_metrics['Cost_per_Test'] = test_metrics['Total_Cost'] / test_metrics['Volume']
    
    # Create scatter plot
    fig = px.scatter(
        test_metrics,
        x='Volume',
        y='Cost_per_Test',
        color='Department',
        size='Total_Cost',
        hover_name='Test_Name',
        title='Volume vs Cost per Test Analysis',
        labels={
            'Volume': 'Total Volume',
            'Cost_per_Test': 'Cost per Test ($)',
            'Total_Cost': 'Total Cost'
        },
        height=600
    )
    
    fig.update_layout(
        xaxis_type="log",
        yaxis_type="log"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.markdown("""
    ### Key Insights:
    - Tests in the upper-left quadrant (high cost, low volume) may be candidates for cost optimization
    - Tests in the lower-right quadrant (low cost, high volume) represent efficient operations
    - Bubble size represents total cost impact
    """)

with analysis_tab3:
    st.subheader("Vendor Concentration Analysis")
    
    # Calculate vendor concentration metrics
    lab_concentration = filtered_df.groupby('Lab').agg({
        'Total_Cost': 'sum',
        'Volume': 'sum',
        'Test_Name': 'nunique',
        'Department': lambda x: len(x.unique())
    }).reset_index()
    
    lab_concentration['Cost_Share'] = lab_concentration['Total_Cost'] / lab_concentration['Total_Cost'].sum() * 100
    lab_concentration['Volume_Share'] = lab_concentration['Volume'] / lab_concentration['Volume'].sum() * 100
    lab_concentration = lab_concentration.sort_values('Cost_Share', ascending=False)
    
    # Calculate concentration ratios
    top_3_concentration = lab_concentration['Cost_Share'].head(3).sum()
    top_5_concentration = lab_concentration['Cost_Share'].head(5).sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Vendor Share Analysis
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Cost Share',
            x=lab_concentration['Lab'].head(10),
            y=lab_concentration['Cost_Share'].head(10),
            text=lab_concentration['Cost_Share'].head(10).apply(lambda x: f'{x:.1f}%'),
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            name='Volume Share',
            x=lab_concentration['Lab'].head(10),
            y=lab_concentration['Volume_Share'].head(10),
            text=lab_concentration['Volume_Share'].head(10).apply(lambda x: f'{x:.1f}%'),
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Top 10 Vendors - Cost vs Volume Share',
            xaxis_tickangle=-45,
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Concentration Metrics
        st.metric("Top 3 Vendor Concentration", f"{top_3_concentration:.1f}%")
        st.metric("Top 5 Vendor Concentration", f"{top_5_concentration:.1f}%")
        
        # Create a pie chart for vendor distribution
        fig = px.pie(
            lab_concentration,
            values='Total_Cost',
            names='Lab',
            title='Vendor Cost Distribution'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # Add vendor diversity insights
    st.markdown("""
    ### Vendor Concentration Insights:
    - High concentration in few vendors may indicate negotiation opportunities
    - Low concentration suggests good vendor diversity but potential for consolidation
    - Consider volume-cost trade-offs when evaluating vendor relationships
    """)

# ABC Analysis
st.header("ABC Analysis")

# Sort data by Total Cost
df_sorted = filtered_df.sort_values(by="Total_Cost", ascending=False).reset_index(drop=True)
total_rows = len(df_sorted)

# Define cutoffs
cutoff_A = int(0.10 * total_rows)
cutoff_B = int(0.30 * total_rows)

# Assign categories
df_sorted["ABC_Category"] = df_sorted.index.map(
    lambda x: "A" if x < cutoff_A else ("B" if x < cutoff_B else "C")
)

# ABC Analysis Summary
abc_summary = df_sorted.groupby("ABC_Category").agg({
    "Total_Cost": ["count", "sum", "mean"],
    "Volume": "sum"
}).round(2)

abc_summary.columns = ["Count", "Total Cost", "Average Cost", "Total Volume"]
abc_summary = abc_summary.reset_index()

# Calculate percentages
abc_summary["Cost %"] = (abc_summary["Total Cost"] / abc_summary["Total Cost"].sum() * 100).round(2)
abc_summary["Volume %"] = (abc_summary["Total Volume"] / abc_summary["Total Volume"].sum() * 100).round(2)

# Display ABC analysis
col1, col2 = st.columns(2)

with col1:
    # Pareto chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=abc_summary, x="ABC_Category", y="Cost %", ax=ax1)
    ax1.set_ylabel("Percentage of Total Cost")
    plt.title("ABC Analysis - Cost Distribution")
    
    st.pyplot(fig)
    plt.close()

with col2:
    st.dataframe(abc_summary.style.format({
        "Count": "{:,.0f}",
        "Total Cost": "${:,.2f}",
        "Average Cost": "${:,.2f}",
        "Total Volume": "{:,.0f}",
        "Cost %": "{:.1f}%",
        "Volume %": "{:.1f}%"
    }))

# Make department selection more prominent
st.markdown("---")
st.markdown("## 📊 Select Department for Detailed Analysis")
st.markdown("### Choose a department below to view detailed metrics and trends")

# Create a selectbox for department selection
selected_department = st.selectbox(
    "Select Department for Detailed Analysis",
    options=[''] + sorted(filtered_df['Department'].unique().tolist()),
    format_func=lambda x: 'Select a department...' if x == '' else x
)

# Show department details if one is selected
if selected_department:
    st.header(f"{selected_department} Department Analysis")
    
    # Filter data for selected department
    dept_data = filtered_df[filtered_df['Department'] == selected_department]
    
    # Department Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Department Total Cost", f"${dept_data['Total_Cost'].sum():,.2f}")
    with col2:
        st.metric("Department Volume", f"{dept_data['Volume'].sum():,.0f}")
    with col3:
        st.metric("Unique Tests", f"{dept_data['Test_Name'].nunique():,}")
    
    # Department Analysis Tabs
    dept_tab1, dept_tab2, dept_tab3 = st.tabs(["Top Tests", "Cost Analysis", "Test Validation & Savings"])
    
    with dept_tab1:
        # Top 10 Tests Table
        st.subheader("Top 10 Tests by Cost")
        top_tests = dept_data.nlargest(10, 'Total_Cost')[
            ['Test_Name', 'Lab', 'Volume', 'Price', 'Total_Cost']
        ].copy()
        
        top_tests['Price'] = top_tests['Price'].apply(lambda x: f"${x:,.2f}")
        top_tests['Total_Cost'] = top_tests['Total_Cost'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            top_tests,
            column_config={
                "Test_Name": st.column_config.TextColumn("Test Name", width="large"),
                "Lab": st.column_config.TextColumn("Lab", width="medium"),
                "Volume": st.column_config.NumberColumn("Volume", format="%d"),
                "Price": st.column_config.TextColumn("Price", width="medium"),
                "Total_Cost": st.column_config.TextColumn("Total Cost", width="medium"),
            },
            hide_index=True,
        )
    
    with dept_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost Distribution
            cost_dist = px.histogram(
                dept_data,
                x='Total_Cost',
                nbins=30,
                title='Cost Distribution',
                labels={'Total_Cost': 'Total Cost ($)'},
                height=400
            )
            st.plotly_chart(cost_dist, use_container_width=True)
        
        with col2:
            # Lab Distribution
            lab_dist = px.pie(
                dept_data.groupby('Lab')['Total_Cost'].sum().reset_index(),
                values='Total_Cost',
                names='Lab',
                title='Cost Distribution by Lab',
                height=400
            )
            st.plotly_chart(lab_dist, use_container_width=True)

    with dept_tab3:
        st.subheader("Test Validation & Cost Savings Analysis")
        
        # Create session state for tracking validation status if it doesn't exist
        if 'validation_status' not in st.session_state:
            st.session_state.validation_status = {}
        
        # Create session state for tracking in-house cost percentages if it doesn't exist
        if 'inhouse_cost_pct' not in st.session_state:
            st.session_state.inhouse_cost_pct = {}
        
        # Auto-suggest tests based on cost and frequency thresholds
        st.markdown("### Auto-Suggest Tests for Validation")
        
        # Create columns for threshold sliders
        threshold_col1, threshold_col2 = st.columns(2)
        
        with threshold_col1:
            cost_threshold = st.slider(
                "Minimum Cost Threshold ($)",
                min_value=0,
                max_value=int(dept_data['Total_Cost'].max()) if not dept_data.empty else 1000,
                value=int(dept_data['Total_Cost'].quantile(0.75)) if not dept_data.empty else 500,
                step=50
            )
        
        with threshold_col2:
            volume_threshold = st.slider(
                "Minimum Volume Threshold",
                min_value=0,
                max_value=int(dept_data['Volume'].max()) if not dept_data.empty else 100,
                value=int(dept_data['Volume'].quantile(0.5)) if not dept_data.empty else 10,
                step=5
            )
        
        # Filter tests based on thresholds
        suggested_tests = dept_data[
            (dept_data['Total_Cost'] >= cost_threshold) &
            (dept_data['Volume'] >= volume_threshold)
        ].sort_values('Total_Cost', ascending=False)
        
        if not suggested_tests.empty:
            st.markdown(f"**{len(suggested_tests)} tests** meet the criteria for potential in-house validation")
            
            # Display suggested tests with validation controls
            st.markdown("### Validation Tracking & Cost Savings")
            
            # Create a dataframe for display with validation status
            display_df = suggested_tests[['Test_Name', 'Lab', 'Volume', 'Price', 'Total_Cost']].copy()
            
            # Add validation status and in-house cost columns
            validation_statuses = []
            inhouse_costs = []
            inhouse_cost_pcts = []
            savings = []
            
            for idx, row in display_df.iterrows():
                test_id = f"{selected_department}_{row['Test_Name']}"
                
                # Get current validation status or default to "Not Started"
                current_status = st.session_state.validation_status.get(test_id, "Not Started")
                
                # Get current in-house cost percentage or default to 60%
                current_pct = st.session_state.inhouse_cost_pct.get(test_id, 60)
                
                validation_statuses.append(current_status)
                inhouse_cost_pcts.append(current_pct)
                
                # Calculate in-house cost and savings
                inhouse_cost = row['Price'] * (current_pct / 100)
                inhouse_costs.append(inhouse_cost)
                
                # Calculate savings based on validation status
                if current_status == "Validated":
                    saving = (row['Price'] - inhouse_cost) * row['Volume']
                else:
                    saving = 0
                    
                savings.append(saving)
            
            display_df['Validation_Status'] = validation_statuses
            display_df['Inhouse_Cost_Pct'] = inhouse_cost_pcts
            display_df['Inhouse_Cost'] = inhouse_costs
            display_df['Potential_Savings'] = savings
            
            # Calculate total potential and actual savings
            total_potential_savings = ((display_df['Price'] - display_df['Inhouse_Cost']) * display_df['Volume']).sum()
            actual_savings = display_df['Potential_Savings'].sum()
            
            # Display savings metrics
            savings_col1, savings_col2 = st.columns(2)
            with savings_col1:
                st.metric(
                    "Total Potential Savings",
                    f"${total_potential_savings:,.2f}",
                    help="Maximum savings if all suggested tests are validated and brought in-house"
                )
            with savings_col2:
                st.metric(
                    "Current Validated Savings",
                    f"${actual_savings:,.2f}",
                    help="Current savings from tests that have been validated and brought in-house"
                )
            
            # Create an editable dataframe
            st.markdown("#### Manage Test Validations")
            st.markdown("Adjust in-house cost percentage and update validation status to calculate savings")
            
            edited_df = st.data_editor(
                display_df,
                column_config={
                    "Test_Name": st.column_config.TextColumn("Test Name", width="large"),
                    "Lab": st.column_config.TextColumn("Lab", width="medium"),
                    "Volume": st.column_config.NumberColumn("Volume", format="%d", width="small"),
                    "Price": st.column_config.NumberColumn("Send-out Price", format="$%.2f", width="medium"),
                    "Total_Cost": st.column_config.NumberColumn("Total Cost", format="$%.2f", width="medium"),
                    "Validation_Status": st.column_config.SelectboxColumn(
                        "Validation Status",
                        options=["Not Started", "In Progress", "Validated", "Failed"],
                        width="medium"
                    ),
                    "Inhouse_Cost_Pct": st.column_config.NumberColumn(
                        "In-house Cost %",
                        min_value=20,
                        max_value=90,
                        step=5,
                        format="%d",
                        width="medium",
                        help="Adjust the percentage (20-90%) of send-out cost for in-house testing"
                    ),
                    "Inhouse_Cost": st.column_config.NumberColumn("In-house Cost", format="$%.2f", width="medium", disabled=True),
                    "Potential_Savings": st.column_config.NumberColumn("Savings", format="$%.2f", width="medium", disabled=True),
                },
                hide_index=True,
                num_rows="fixed",
                key=f"validation_table_{selected_department}"
            )
            
            # Update session state based on edited values
            for idx, row in edited_df.iterrows():
                test_id = f"{selected_department}_{row['Test_Name']}"
                st.session_state.validation_status[test_id] = row['Validation_Status']
                st.session_state.inhouse_cost_pct[test_id] = row['Inhouse_Cost_Pct']
            
            # Visualization of validation progress
            st.markdown("### Validation Progress")
            
            # Count tests by validation status
            status_counts = edited_df['Validation_Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            # Create a pie chart of validation status
            fig = px.pie(
                status_counts,
                values='Count',
                names='Status',
                title='Validation Status Distribution',
                color='Status',
                color_discrete_map={
                    'Not Started': '#d3d3d3',
                    'In Progress': '#ffa500',
                    'Validated': '#2e8b57',
                    'Failed': '#dc143c'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide recommendations based on validation status
            st.markdown("### Recommendations")
            
            # Tests with highest potential savings that are not yet validated
            high_savings_tests = edited_df[
                (edited_df['Validation_Status'] != "Validated") & 
                (edited_df['Validation_Status'] != "Failed")
            ].copy()
            
            if not high_savings_tests.empty:
                high_savings_tests['Potential_Savings'] = (high_savings_tests['Price'] - high_savings_tests['Inhouse_Cost']) * high_savings_tests['Volume']
                high_savings_tests = high_savings_tests.sort_values('Potential_Savings', ascending=False)
                
                st.markdown("#### Priority Tests for Validation")
                st.markdown("These tests have the highest potential savings and should be prioritized for validation:")
                
                for _, row in high_savings_tests.head(3).iterrows():
                    st.markdown(f"""
                    - **{row['Test_Name']}** from {row['Lab']}
                      - Current send-out cost: ${row['Price']:.2f} per test
                      - Potential in-house cost: ${row['Inhouse_Cost']:.2f} per test
                      - Annual volume: {row['Volume']:.0f} tests
                      - Potential annual savings: ${row['Potential_Savings']:.2f}
                    """)
        else:
            st.warning("No tests meet the current threshold criteria. Try adjusting the thresholds.")

# Top Tests Analysis
st.header("Top Tests Analysis")
top_n = st.slider("Select number of top tests to view", 5, 50, 10)

top_tests = df_sorted.head(top_n)[['Test_Name', 'Department', 'Lab', 'Volume', 'Price', 'Total_Cost']]
st.dataframe(top_tests.style.format({
    "Volume": "{:,.0f}",
    "Price": "${:,.2f}",
    "Total_Cost": "${:,.2f}"
}))

# Cost Savings Opportunities
st.header("Cost Savings Opportunities")

# Calculate potential savings opportunities
opportunities = []

# 1. High-cost, low-volume tests
high_cost_low_volume = test_metrics[
    (test_metrics['Cost_per_Test'] > test_metrics['Cost_per_Test'].quantile(0.75)) &
    (test_metrics['Volume'] < test_metrics['Volume'].quantile(0.25))
].copy()
if not high_cost_low_volume.empty:
    opportunities.append({
        'category': 'High Cost, Low Volume Tests',
        'description': 'Tests with high unit costs and low volumes - consider insourcing or negotiating prices',
        'tests': high_cost_low_volume[['Test_Name', 'Department', 'Lab', 'Cost_per_Test', 'Volume', 'Total_Cost']]
    })

# 2. Vendor consolidation opportunities
dept_lab_counts = filtered_df.groupby('Department')['Lab'].nunique()
high_vendor_depts = dept_lab_counts[dept_lab_counts > dept_lab_counts.median()].index
vendor_consolidation = filtered_df[filtered_df['Department'].isin(high_vendor_depts)].groupby(['Department', 'Lab']).agg({
    'Total_Cost': 'sum',
    'Test_Name': 'nunique'
}).reset_index()
if not vendor_consolidation.empty:
    opportunities.append({
        'category': 'Vendor Consolidation',
        'description': 'Departments with high vendor fragmentation - potential for consolidation',
        'data': vendor_consolidation
    })

# 3. Price variation analysis
test_price_variation = filtered_df.groupby('Test_Name').agg({
    'Total_Cost': ['mean', 'std', 'count']
}).reset_index()
test_price_variation.columns = ['Test_Name', 'Mean_Cost', 'Std_Cost', 'Count']
test_price_variation['CV'] = test_price_variation['Std_Cost'] / test_price_variation['Mean_Cost']
high_variation_tests = test_price_variation[
    (test_price_variation['CV'] > 0.5) & 
    (test_price_variation['Count'] > 5)
].copy()
if not high_variation_tests.empty:
    opportunities.append({
        'category': 'Price Standardization',
        'description': 'Tests with high price variation - opportunity for standardization',
        'tests': high_variation_tests
    })

# Display opportunities
for opp in opportunities:
    with st.expander(f"💡 {opp['category']}", expanded=True):
        st.write(opp['description'])
        if 'tests' in opp:
            st.dataframe(
                opp['tests'].style.format({
                    'Cost_per_Test': '${:,.2f}',
                    'Total_Cost': '${:,.2f}',
                    'Volume': '{:,.0f}',
                    'CV': '{:.2f}'
                }),
                hide_index=True
            )
        elif 'data' in opp:
            st.dataframe(
                opp['data'].style.format({
                    'Total_Cost': '${:,.2f}',
                    'Test_Name': '{:,.0f}'
                }),
                hide_index=True
            )

# Export and Reporting
st.header("Export & Reports")

col1, col2 = st.columns(2)

with col1:
    # Excel Export
    st.subheader("Export Data")
    
    # Create Excel file in memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        # Main data
        filtered_df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        # Summary sheets
        dept_efficiency.to_excel(writer, sheet_name='Department Analysis', index=False)
        test_metrics.to_excel(writer, sheet_name='Test Metrics', index=False)
        lab_concentration.to_excel(writer, sheet_name='Vendor Analysis', index=False)
        abc_summary.to_excel(writer, sheet_name='ABC Analysis', index=False)
        
        # Opportunities
        if not high_cost_low_volume.empty:
            high_cost_low_volume.to_excel(writer, sheet_name='Cost Opportunities', index=False)
        
        # Add validation data if available
        if selected_department and 'validation_status' in st.session_state and 'suggested_tests' in locals() and not suggested_tests.empty:
            # Create a validation summary dataframe
            validation_summary = []
            
            display_df = suggested_tests[['Test_Name', 'Lab', 'Volume', 'Price', 'Total_Cost']].copy()
            
            for idx, row in display_df.iterrows():
                test_id = f"{selected_department}_{row['Test_Name']}"
                
                # Get current validation status or default to "Not Started"
                current_status = st.session_state.validation_status.get(test_id, "Not Started")
                
                # Get current in-house cost percentage or default to 60%
                current_pct = st.session_state.inhouse_cost_pct.get(test_id, 60)
                
                # Calculate in-house cost and savings
                inhouse_cost = row['Price'] * (current_pct / 100)
                
                # Calculate savings
                potential_saving = (row['Price'] - inhouse_cost) * row['Volume']
                actual_saving = potential_saving if current_status == "Validated" else 0
                
                validation_summary.append({
                    'Department': selected_department,
                    'Test_Name': row['Test_Name'],
                    'Lab': row['Lab'],
                    'Volume': row['Volume'],
                    'Send_Out_Price': row['Price'],
                    'Total_Cost': row['Total_Cost'],
                    'Validation_Status': current_status,
                    'Inhouse_Cost_Pct': current_pct,
                    'Inhouse_Cost': inhouse_cost,
                    'Potential_Savings': potential_saving,
                    'Actual_Savings': actual_saving
                })
            
            if validation_summary:
                validation_df = pd.DataFrame(validation_summary)
                validation_df.to_excel(writer, sheet_name='Validation Summary', index=False)
    
    excel_buffer.seek(0)
    st.download_button(
        label="📥 Download Full Analysis (Excel)",
        data=excel_buffer,
        file_name=f"send_out_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.ms-excel"
    )

with col2:
    # PDF Report Generation
    st.subheader("Generate Report")
    
    if st.button("📄 Generate PDF Report"):
        # Create PDF with more details
        pdf = FPDF()
        pdf.set_font('helvetica', '', 12)  # Use helvetica instead of arial
        
        # Import FPDF constants
        from fpdf import XPos, YPos
        
        # Title Page
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 24)
        pdf.cell(0, 20, 'Send-out Test Analysis Report', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('helvetica', '', 12)
        pdf.cell(0, 10, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Executive Summary
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, 'Executive Summary', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_font('helvetica', '', 12)
        pdf.multi_cell(0, 10, 'This report provides a comprehensive analysis of medical test send-out data, including cost patterns, vendor relationships, and potential cost-saving opportunities.')
        
        # Key Metrics
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, 'Key Metrics', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('helvetica', '', 12)
        pdf.cell(0, 10, f"Total Tests Analyzed: {len(filtered_df):,}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 10, f"Total Cost: ${filtered_df['Total_Cost'].sum():,.2f}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 10, f"Average Cost per Test: ${filtered_df['Total_Cost'].mean():,.2f}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 10, f"Number of Departments: {filtered_df['Department'].nunique()}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 10, f"Number of Vendors: {filtered_df['Lab'].nunique()}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # ABC Analysis
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, 'ABC Analysis', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_font('helvetica', '', 12)
        
        for _, row in abc_summary.iterrows():
            pdf.multi_cell(0, 10, 
                    f"Category {row['ABC_Category']}:\n" +
                    f"- {row['Count']:.0f} tests ({row['Count']/abc_summary['Count'].sum()*100:.1f}% of total)\n" +
                    f"- ${row['Total Cost']:,.2f} total cost ({row['Cost %']:.1f}% of spend)\n" +
                    f"- {row['Total Volume']:,.0f} total volume ({row['Volume %']:.1f}% of volume)")
            pdf.set_y(pdf.get_y() + 5)
        
        # Department Analysis
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, 'Department Analysis', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_y(pdf.get_y() + 5)
        
        # Top 5 departments by cost
        top_5_dept = dept_efficiency.nlargest(5, 'Total_Cost')
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, 'Top 5 Departments by Cost', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('helvetica', '', 12)
        for _, row in top_5_dept.iterrows():
            pdf.multi_cell(0, 10, 
                    f"{row['Department']}:\n" +
                    f"- Total Cost: ${row['Total_Cost']:,.2f}\n" +
                    f"- Cost per Test: ${row['Cost_per_Test']:,.2f}\n" +
                    f"- Volume: {row['Volume']:,.0f}")
            pdf.set_y(pdf.get_y() + 5)
        
        # Vendor Analysis
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, 'Vendor Analysis', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_font('helvetica', '', 12)
        pdf.multi_cell(0, 10, 
                f"Top 3 Vendor Concentration: {top_3_concentration:.1f}%\n" +
                f"Top 5 Vendor Concentration: {top_5_concentration:.1f}%")
        
        # Top 5 vendors
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, 'Top 5 Vendors by Cost', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('helvetica', '', 12)
        top_5_vendors = lab_concentration.head()
        for _, row in top_5_vendors.iterrows():
            pdf.multi_cell(0, 10,
                    f"{row['Lab']}:\n" +
                    f"- Cost Share: {row['Cost_Share']:.1f}%\n" +
                    f"- Volume Share: {row['Volume_Share']:.1f}%\n" +
                    f"- Number of Tests: {row['Test_Name']:.0f}")
            pdf.set_y(pdf.get_y() + 5)
        
        # Cost Saving Opportunities
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, 'Cost Saving Opportunities', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_font('helvetica', '', 12)
        
        for opp in opportunities:
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, f"- {opp['category']}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('helvetica', 'I', 12)
            pdf.multi_cell(0, 10, opp['description'])
            if 'tests' in opp and not opp['tests'].empty:
                pdf.set_y(pdf.get_y() + 5)
                pdf.set_font('helvetica', '', 12)
                top_3_tests = opp['tests'].head(3)
                pdf.multi_cell(0, 10, "Top 3 Examples:")
                for _, test in top_3_tests.iterrows():
                    pdf.multi_cell(0, 10, 
                            f"- {test['Test_Name']}\n" +
                            f"  Cost: ${test.get('Cost_per_Test', 0):,.2f} per test\n" +
                            f"  Volume: {test.get('Volume', 0):,.0f}")
                pdf.set_y(pdf.get_y() + 5)
        
        # Add Test Validation Summary if a department is selected
        if selected_department and 'validation_status' in st.session_state:
            # Get all tests for the selected department that have validation status
            validated_tests = []
            
            # Check if there are any suggested tests with validation status
            if 'suggested_tests' in locals() and not suggested_tests.empty:
                display_df = suggested_tests[['Test_Name', 'Lab', 'Volume', 'Price', 'Total_Cost']].copy()
                
                # Add validation status and in-house cost columns
                for idx, row in display_df.iterrows():
                    test_id = f"{selected_department}_{row['Test_Name']}"
                    
                    # Get current validation status or default to "Not Started"
                    current_status = st.session_state.validation_status.get(test_id, "Not Started")
                    
                    # Get current in-house cost percentage or default to 60%
                    current_pct = st.session_state.inhouse_cost_pct.get(test_id, 60)
                    
                    # Calculate in-house cost and savings
                    inhouse_cost = row['Price'] * (current_pct / 100)
                    
                    # Calculate savings based on validation status
                    if current_status == "Validated":
                        saving = (row['Price'] - inhouse_cost) * row['Volume']
                        validated_tests.append({
                            'Test_Name': row['Test_Name'],
                            'Lab': row['Lab'],
                            'Volume': row['Volume'],
                            'Price': row['Price'],
                            'Inhouse_Cost': inhouse_cost,
                            'Savings': saving
                        })
            
            if validated_tests:
                pdf.add_page()
                pdf.set_font('helvetica', 'B', 16)
                pdf.cell(0, 10, f'Test Validation Summary - {selected_department}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_y(pdf.get_y() + 5)
                
                # Calculate total savings
                total_savings = sum(test['Savings'] for test in validated_tests)
                
                pdf.set_font('helvetica', 'B', 14)
                pdf.cell(0, 10, f"Total Validated Savings: ${total_savings:,.2f}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_y(pdf.get_y() + 5)
                
                pdf.set_font('helvetica', 'B', 14)
                pdf.cell(0, 10, f"Validated Tests ({len(validated_tests)})", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('helvetica', '', 12)
                
                for test in validated_tests:
                    pdf.multi_cell(0, 10, 
                            f"- {test['Test_Name']} (from {test['Lab']})\n" +
                            f"  Send-out Cost: ${test['Price']:,.2f} per test\n" +
                            f"  In-house Cost: ${test['Inhouse_Cost']:,.2f} per test\n" +
                            f"  Annual Volume: {test['Volume']:,.0f} tests\n" +
                            f"  Annual Savings: ${test['Savings']:,.2f}")
                    pdf.set_y(pdf.get_y() + 5)
        
        # Save PDF to buffer
        pdf_buffer = io.BytesIO()
        pdf_buffer.write(pdf.output())
        pdf_buffer.seek(0)
        
        # Offer download
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"send_out_report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit - Medical Test Send-out Analysis")
