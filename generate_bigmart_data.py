"""
BigMart-like dataset generator using Faker

Creates a synthetic dataset with similar structure to the BigMart sales data:
- Item/product attributes
- Outlet/store characteristics
- Sales performance metrics
"""

from faker import Faker
import pandas as pd
import numpy as np
from random import choices, randint, uniform

# Initialize Faker with seed for reproducibility
fake = Faker()
Faker.seed(42)

# Configuration
NUM_RECORDS = 10_000  # Generate 10k records

# Define possible values mirroring BigMart data
ITEM_TYPES = [
    "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
    "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Soft Drinks",
    "Meat", "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast",
    "Seafood"
]

OUTLET_TYPES = ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
OUTLET_SIZES = ["Small", "Medium", "High"]
OUTLET_LOCATIONS = ["Tier 1", "Tier 2", "Tier 3"]
FAT_CONTENT = ["low", "regular"]

# Helper functions
def generate_item_identifier():
    return f"FD{fake.random_int(min=100, max=999)}{fake.random_letter().upper()}"

def generate_outlet_identifier():
    return f"OUT{fake.random_int(min=10, max=99)}{fake.random_letter().upper()}"

def generate_item_weight():
    return round(uniform(4.5, 21.5), 2)

def generate_mrp():
    return round(uniform(30, 270), 2)

def generate_visibility():
    return round(uniform(0.01, 0.4), 2)

def generate_sales(outlet_type, outlet_size):
    """Generate sales based on outlet characteristics"""
    base = {"Grocery Store": 800, "Supermarket Type1": 1200, 
           "Supermarket Type2": 1800, "Supermarket Type3": 2500}
    size_mult = {"Small": 0.8, "Medium": 1.0, "High": 1.3}
    
    return round(base[outlet_type] * size_mult[outlet_size] * uniform(0.7, 1.3), 2)

# Generate dataset
data = []
for _ in range(NUM_RECORDS):
    # Item attributes
    item_type = fake.random_element(ITEM_TYPES)
    fat_content = fake.random_element(FAT_CONTENT)
    
    # Skip fat content for non-food items
    if item_type in ["Household", "Health and Hygiene"]:
        fat_content = "NA"
    
    # Outlet attributes
    outlet_type = fake.random_element(OUTLET_TYPES)
    outlet_size = fake.random_element(OUTLET_SIZES)
    outlet_loc = fake.random_element(OUTLET_LOCATIONS)
    
    record = {
        "Item_Identifier": generate_item_identifier(),
        "Item_Weight": generate_item_weight(),
        "Item_Fat_Content": fat_content,
        "Item_Visibility": generate_visibility(),
        "Item_Type": item_type,
        "Item_MRP": generate_mrp(),
        "Outlet_Identifier": generate_outlet_identifier(),
        "Outlet_Establishment_Year": fake.random_int(min=1985, max=2009),
        "Outlet_Size": outlet_size,
        "Outlet_Location_Type": outlet_loc,
        "Outlet_Type": outlet_type,
        "Item_Outlet_Sales": generate_sales(outlet_type, outlet_size)
    }
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values (5% chance)
for col in ["Item_Weight", "Outlet_Size"]:
    mask = np.random.choice([True, False], size=len(df), p=[0.05, 0.95])
    df.loc[mask, col] = np.nan

# Save to CSV
output_path = "bigmart_synthetic_data.csv"
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} records saved to {output_path}")
print(df.head())
