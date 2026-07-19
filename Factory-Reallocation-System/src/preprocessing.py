import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956 # Radius of earth in miles
    return c * r

def main():
    print("Loading data...")
    # Load data
    orders = pd.read_csv('../data/raw/orders.csv')
    orders['Order Date'] = pd.to_datetime(orders['Order Date'], format='%d-%m-%Y')
    
    coords = pd.read_csv('../data/raw/factory_coordinates.csv')
    mapping = pd.read_csv('../data/raw/product_factory_mapping.csv')
    
    # Ignore existing Ship Date
    if 'Ship Date' in orders.columns:
        orders = orders.drop(columns=['Ship Date'])
    
    # Join factory via Product Name
    orders = orders.merge(mapping[['Product Name', 'Factory']], on='Product Name', how='left')
    
    # Join factory coordinates
    orders = orders.merge(coords, on='Factory', how='left')
    
    # Approximate region coordinates
    region_coords = {
        'Interior': (40.0, -95.0),
        'Atlantic': (39.0, -76.0),
        'Gulf': (30.0, -90.0),
        'Pacific': (37.0, -120.0)
    }
    
    def get_region_lat(region):
        return region_coords.get(region, (39.8283, -98.5795))[0]
    
    def get_region_lon(region):
        return region_coords.get(region, (39.8283, -98.5795))[1]
        
    orders['Region_Lat'] = orders['Region'].apply(get_region_lat)
    orders['Region_Lon'] = orders['Region'].apply(get_region_lon)
    
    # Calculate distance
    orders['distance'] = haversine(
        orders['Latitude'], orders['Longitude'],
        orders['Region_Lat'], orders['Region_Lon']
    )
    
    # Lead time base calculation
    np.random.seed(42)
    def base_lead_time(mode):
        if mode == 'Same Day':
            return np.random.uniform(0, 1)
        elif mode == 'First Class':
            return np.random.uniform(1, 3)
        elif mode == 'Second Class':
            return np.random.uniform(3, 5)
        else: # Standard Class
            return np.random.uniform(5, 9)
            
    orders['base_days'] = orders['Ship Mode'].apply(base_lead_time)
    
    # Distance scaling: 1 day per 500 miles
    distance_days = orders['distance'] / 500.0
    
    # Random noise (-1 to 2 days)
    noise_days = np.random.uniform(-1, 2, size=len(orders))
    
    orders['lead_time_days'] = np.round(orders['base_days'] + distance_days + noise_days).astype(int)
    
    # Clip extreme outliers (ensure lead time is not negative)
    orders['lead_time_days'] = orders['lead_time_days'].clip(lower=0)
    
    # Calculate new Ship Date
    orders['Ship Date'] = orders['Order Date'] + pd.to_timedelta(orders['lead_time_days'], unit='D')
    
    # Remove duplicates
    orders = orders.drop_duplicates()
    
    # Save to processed
    orders.to_csv('../data/processed/clean_orders.csv', index=False)
    
    # Print sample
    sample_cols = ['Order Date', 'Ship Mode', 'Region', 'Factory', 'distance', 'lead_time_days', 'Ship Date']
    print("\nSample of 15 rows:")
    print(orders[sample_cols].head(15).to_string(index=False))
    
    print("\nSummary: Average lead time days by Factory and Ship Mode:")
    summary = orders.groupby(['Factory', 'Ship Mode'])['lead_time_days'].mean().reset_index()
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
