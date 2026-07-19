import pandas as pd
import numpy as np
import joblib
import os

# Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956 # Radius of earth in miles
    return c * r

# Global variables to cache data and models
_model = None
_encoders = None
_feature_cols = None
_df = None
_factory_coords = None
_region_coords = None

def init_simulator():
    global _model, _encoders, _feature_cols, _df, _factory_coords, _region_coords
    if _model is not None:
        return # already loaded
        
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'clean_orders.csv')
    
    _model = joblib.load(os.path.join(models_dir, 'best_model.joblib'))
    _encoders = joblib.load(os.path.join(models_dir, 'encoders.joblib'))
    
    # Try loading feature columns if exists, otherwise hardcode
    try:
        _feature_cols = joblib.load(os.path.join(models_dir, 'feature_columns.joblib'))
    except FileNotFoundError:
        _feature_cols = ['Product Name', 'Factory', 'Region', 'Ship Mode', 'Sales', 'Units', 'Cost', 'Gross Profit', 'distance']
        
    _df = pd.read_csv(data_path)
    
    # Extract unique factory coordinates
    _factory_coords = _df[['Factory', 'Latitude', 'Longitude']].drop_duplicates().set_index('Factory')
    
    # Extract unique region coordinates
    _region_coords = _df[['Region', 'Region_Lat', 'Region_Lon']].drop_duplicates().set_index('Region')

def simulate_product(product_name):
    init_simulator()
    
    prod_df = _df[_df['Product Name'] == product_name]
    if prod_df.empty:
        raise ValueError(f"Product '{product_name}' not found in data.")
        
    # Look up current factory
    current_factory = prod_df['Factory'].iloc[0]
    
    # Average metrics
    avg_sales = prod_df['Sales'].mean()
    avg_units = prod_df['Units'].mean()
    avg_cost = prod_df['Cost'].mean()
    avg_gross_profit = prod_df['Gross Profit'].mean()
    
    # Most frequent ship mode
    ship_mode = prod_df['Ship Mode'].mode()[0]
    
    # Regions it ships to
    regions = prod_df['Region'].unique()
    
    all_factories = _factory_coords.index.tolist()
    
    results = []
    
    for candidate_factory in all_factories:
        for region in regions:
            fac_lat = _factory_coords.loc[candidate_factory, 'Latitude']
            fac_lon = _factory_coords.loc[candidate_factory, 'Longitude']
            reg_lat = _region_coords.loc[region, 'Region_Lat']
            reg_lon = _region_coords.loc[region, 'Region_Lon']
            
            # Compute distance
            dist = haversine(fac_lat, fac_lon, reg_lat, reg_lon)
            
            # Prepare row
            row_dict = {
                'Product Name': product_name,
                'Factory': candidate_factory,
                'Region': region,
                'Ship Mode': ship_mode,
                'Sales': avg_sales,
                'Units': avg_units,
                'Cost': avg_cost,
                'Gross Profit': avg_gross_profit,
                'distance': dist
            }
            results.append(row_dict)
            
    # DataFrame for prediction
    hypo_df = pd.DataFrame(results)
    
    # Current distance for each region to calculate risk flag
    current_dist_map = hypo_df[hypo_df['Factory'] == current_factory].set_index('Region')['distance'].to_dict()
    
    # Encode categorical features
    X = hypo_df[_feature_cols].copy()
    for col, le in _encoders.items():
        if col in X.columns:
            # Handle unknown classes by assigning a default or fallback (not expected here since we use known products/factories/regions)
            X[col] = le.transform(X[col])
            
    # Predict
    hypo_df['Predicted Lead Time'] = _model.predict(X)
    
    # Current factory predicted lead times map
    current_lt_map = hypo_df[hypo_df['Factory'] == current_factory].set_index('Region')['Predicted Lead Time'].to_dict()
    
    # Calculate comparisons
    hypo_df['Current Lead Time'] = hypo_df['Region'].map(current_lt_map)
    hypo_df['Lead Time Change vs Current (%)'] = ((hypo_df['Predicted Lead Time'] - hypo_df['Current Lead Time']) / hypo_df['Current Lead Time']) * 100
    
    hypo_df['Current Distance'] = hypo_df['Region'].map(current_dist_map)
    hypo_df['Risk Flag'] = hypo_df.apply(lambda row: "High Logistics Risk" if row['distance'] > 1.5 * row['Current Distance'] else "OK", axis=1)
    
    # Final ranked table per region
    final_cols = ['Region', 'Factory', 'Predicted Lead Time', 'Lead Time Change vs Current (%)', 'distance', 'Risk Flag']
    final_df = hypo_df[final_cols].rename(columns={'distance': 'Distance'}).copy()
    
    # Add a column indicating if this is the current factory
    final_df['Is Current'] = final_df['Factory'] == current_factory
    
    # Sort by Region, then by Predicted Lead Time ascending
    final_df = final_df.sort_values(['Region', 'Predicted Lead Time'], ascending=[True, True])
    
    return final_df

def main():
    print("Initializing Scenario Simulation Engine...\n")
    init_simulator()
    
    # Sample products (Chocolate, Sugar, Other)
    # Get one product from each division
    divisions = _df['Division'].unique()
    sample_products = []
    for div in ['Chocolate', 'Sugar', 'Other']:
        if div in _df['Division'].values:
            prod = _df[_df['Division'] == div]['Product Name'].iloc[0]
            sample_products.append(prod)
    
    # If any divisions are missing, just pick random ones
    if not sample_products:
        sample_products = _df['Product Name'].drop_duplicates().sample(3, random_state=42).tolist()
        
    for prod in sample_products:
        print("="*80)
        print(f"Simulation for Product: {prod}")
        print("="*80)
        
        result = simulate_product(prod)
        
        # Display nicely grouped by region
        for region, group in result.groupby('Region', sort=False): # sort=False keeps the previous sorting (which is by predicted lead time)
            print(f"\n--- Region: {region} ---")
            display_df = group[['Factory', 'Is Current', 'Predicted Lead Time', 'Lead Time Change vs Current (%)', 'Distance', 'Risk Flag']]
            display_df = display_df.copy()
            # Format numbers
            display_df['Predicted Lead Time'] = display_df['Predicted Lead Time'].round(2)
            display_df['Lead Time Change vs Current (%)'] = display_df['Lead Time Change vs Current (%)'].round(2).astype(str) + '%'
            display_df['Distance'] = display_df['Distance'].round(2)
            print(display_df.to_string(index=False))
        print("\n")

if __name__ == "__main__":
    main()
