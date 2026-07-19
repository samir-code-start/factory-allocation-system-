import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

def main():
    print("Loading data...")
    # 1. Load data
    data_path = 'data/processed/clean_orders.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    
    print("Aggregating data...")
    # 2. Group by Factory, Region, Ship Mode
    # Aggregate avg lead_time_days, avg distance, and count of Order ID
    grouped = df.groupby(['Factory', 'Region', 'Ship Mode']).agg(
        lead_time_days=('lead_time_days', 'mean'),
        distance=('distance', 'mean'),
        order_count=('Order ID', 'count')
    ).reset_index()
    
    print("Standardizing features...")
    # 3. Standardize distance and lead_time_days
    features = ['distance', 'lead_time_days']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(grouped[features])
    
    print("Running K-Means...")
    # 4. Run K-Means with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    grouped['cluster_id'] = kmeans.fit_predict(scaled_features)
    
    print("Labeling clusters...")
    # 5. Label each cluster based on actual centroid values
    # The prompt says: "derive the label from whether that cluster's average lead_time_days and order volume are high or low relative to the others."
    cluster_stats = grouped.groupby('cluster_id').agg(
        avg_lead_time=('lead_time_days', 'mean'),
        avg_order_volume=('order_count', 'mean')
    ).reset_index()
    
    # Calculate median to determine high/low
    median_lead_time = cluster_stats['avg_lead_time'].median()
    median_volume = cluster_stats['avg_order_volume'].median()
    
    def assign_label(row):
        is_high_lead_time = row['avg_lead_time'] >= median_lead_time
        is_high_volume = row['avg_order_volume'] >= median_volume
        
        if is_high_volume and not is_high_lead_time:
            return "High-Performing Regions"
        elif is_high_volume and is_high_lead_time:
            return "Congested Regions"
        elif not is_high_volume and not is_high_lead_time:
            return "Fast Routes"
        else: # not high volume and high lead time
            return "Slow Routes"
            
    cluster_stats['cluster_label'] = cluster_stats.apply(assign_label, axis=1)
    
    # Ensure unique labels if the median split results in duplicates (e.g. if medians don't neatly divide 4 clusters into 4 categories)
    # A more robust way to ensure 1 of each of the 4 labels:
    # Sort clusters by order volume (descending) -> Top 2 are High Volume, Bottom 2 are Low Volume
    # Then within each volume group, sort by lead time (descending) -> Top 1 is High Lead Time, Bottom 1 is Low Lead Time
    
    # Let's do the robust mapping:
    cluster_stats = cluster_stats.sort_values('avg_order_volume', ascending=False)
    high_vol_clusters = cluster_stats.iloc[:2].copy()
    low_vol_clusters = cluster_stats.iloc[2:].copy()
    
    high_vol_clusters = high_vol_clusters.sort_values('avg_lead_time', ascending=False)
    high_vol_clusters['cluster_label'] = ["Congested Regions", "High-Performing Regions"]
    
    low_vol_clusters = low_vol_clusters.sort_values('avg_lead_time', ascending=False)
    low_vol_clusters['cluster_label'] = ["Slow Routes", "Fast Routes"]
    
    final_labels = pd.concat([high_vol_clusters, low_vol_clusters])[['cluster_id', 'cluster_label']]
    
    # Merge labels back
    grouped = grouped.merge(final_labels, on='cluster_id')
    grouped = grouped.drop('cluster_id', axis=1) # Drop the numeric ID as we have the label
    
    print("\nCluster Assignments:")
    # 6. Print readable table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(grouped.to_string(index=False))
    pd.reset_option('display.max_rows')
    
    print("\nSaving results...")
    # 7. Save results and models
    os.makedirs('data/processed', exist_ok=True)
    grouped.to_csv('data/processed/route_clusters.csv', index=False)
    
    os.makedirs('models', exist_ok=True)
    model_data = {
        'kmeans': kmeans,
        'scaler': scaler,
        'labels_map': final_labels.set_index('cluster_id')['cluster_label'].to_dict()
    }
    joblib.dump(model_data, 'models/route_clustering.joblib')
    print("Done. Saved to data/processed/route_clusters.csv and models/route_clustering.joblib")

if __name__ == "__main__":
    main()
