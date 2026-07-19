import nbformat as nbf
import os
import subprocess

def main():
    print("Creating notebooks/03_clustering.ipynb...")
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Factory Route & Product Clustering\n\nThis notebook analyzes the factory allocation routes by grouping them based on lead time and distance."),
        
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the clustered data
df_clusters = pd.read_csv('../data/processed/route_clusters.csv')
df_clusters.head()"""),
        
        nbf.v4.new_markdown_cell("## Cluster Visualization\n\nVisualizing the route clusters based on average distance and lead time days."),
        
        nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_clusters, 
    x='distance', 
    y='lead_time_days', 
    hue='cluster_label',
    palette='Set1',
    s=100,
    alpha=0.7
)

plt.title('Route Clusters: Distance vs Lead Time Days')
plt.xlabel('Average Distance')
plt.ylabel('Average Lead Time (Days)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("## Cluster Profiles\n\nSummary of statistics for each route cluster."),

        nbf.v4.new_code_cell("""cluster_summary = df_clusters.groupby('cluster_label').agg(
    avg_lead_time=('lead_time_days', 'mean'),
    avg_distance=('distance', 'mean'),
    total_orders=('order_count', 'sum'),
    route_count=('cluster_label', 'count')
).reset_index()

cluster_summary""")
    ]
    
    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/03_clustering.ipynb', 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook created.")

if __name__ == "__main__":
    main()
