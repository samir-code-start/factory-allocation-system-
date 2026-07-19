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
    
    # Rank all 4 clusters directly by avg lead_time_days first (ascending = faster)
    cluster_stats = cluster_stats.sort_values('avg_lead_time', ascending=True)
    
    # Split into fast half (top 2) and slow half (bottom 2) based strictly on lead time
    fast_half = cluster_stats.iloc[:2].copy()
    slow_half = cluster_stats.iloc[2:].copy()
    
    # Within fast half, use volume as a secondary descriptor
    fast_half = fast_half.sort_values('avg_order_volume', ascending=False)
    fast_half['cluster_label'] = ["High-Performing Regions", "Fast Routes"]
    
    # Within slow half, use volume as a secondary descriptor
    slow_half = slow_half.sort_values('avg_order_volume', ascending=False)
    slow_half['cluster_label'] = ["Congested Regions", "Slow Routes"]
    
    final_labels = pd.concat([fast_half, slow_half])[['cluster_id', 'cluster_label']]
    
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
