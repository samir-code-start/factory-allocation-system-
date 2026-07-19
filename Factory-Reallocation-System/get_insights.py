import pandas as pd

df = pd.read_csv('data/processed/clean_orders.csv')

print("--- Sales by Region ---")
print(df.groupby('Region')['Sales'].sum().sort_values(ascending=False))

print("\n--- Profit by Region ---")
profit_by_region = df.groupby('Region').agg({'Sales': 'sum', 'Gross Profit': 'sum'}).reset_index()
profit_by_region['Profit Margin'] = profit_by_region['Gross Profit'] / profit_by_region['Sales']
print(profit_by_region.sort_values(by='Profit Margin'))

print("\n--- Sales by Factory ---")
print(df.groupby('Factory')['Sales'].sum().sort_values(ascending=False))

print("\n--- Factory Utilization ---")
print(df.groupby('Factory').agg({'Order ID': 'nunique', 'Units': 'sum'}))

print("\n--- Ship Mode ---")
print(df['Ship Mode'].value_counts())

print("\n--- Top Products by Profit ---")
print(df.groupby('Product Name')['Gross Profit'].sum().sort_values(ascending=False).head(5))

print("\n--- Bottom Products by Profit ---")
print(df.groupby('Product Name')['Gross Profit'].sum().sort_values(ascending=False).tail(5))

print("\n--- Lead Time by Factory ---")
print(df.groupby('Factory')['lead_time_days'].mean().sort_values())
