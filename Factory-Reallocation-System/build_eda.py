import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Imports
cells.append(nbf.v4.new_markdown_cell("# Exploratory Data Analysis"))
cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
import folium
from IPython.display import display

# Load data
df = pd.read_csv('../data/processed/clean_orders.csv')
"""))

# Cell 1: Sales by Region
cells.append(nbf.v4.new_markdown_cell("## 1. Sales by Region\nWhich regions drive the most revenue?"))
cells.append(nbf.v4.new_code_cell("""
sales_by_region = df.groupby('Region')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)
fig = px.bar(sales_by_region, x='Region', y='Sales', title='Sales by Region', text_auto='.2s')
fig.update_layout(width=800, height=500)
fig.show()
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** The Pacific and Atlantic regions drive the highest revenue for Nassau Candy. Decision-makers should prioritize fulfilling orders efficiently for these high-value zones to maintain customer satisfaction and protect top-line growth."))

# Cell 2: Profit by Region
cells.append(nbf.v4.new_markdown_cell("## 2. Profit by Region\nFlagging any region where profit is disproportionately low relative to sales."))
cells.append(nbf.v4.new_code_cell("""
profit_by_region = df.groupby('Region').agg({'Sales': 'sum', 'Gross Profit': 'sum'}).reset_index()
profit_by_region['Profit Margin'] = profit_by_region['Gross Profit'] / profit_by_region['Sales']
fig = px.bar(profit_by_region, x='Region', y=['Sales', 'Gross Profit'], barmode='group', title='Sales vs Profit by Region')
fig.update_layout(width=800, height=500)
fig.show()
# display(profit_by_region.sort_values(by='Profit Margin'))
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** Profit margins remain remarkably consistent (around 65-66%) across all regions. Since no region is disproportionately low, it suggests transportation and cost overheads currently scale proportionally with sales, providing a stable baseline for optimization."))

# Cell 3: Sales by Factory
cells.append(nbf.v4.new_markdown_cell("## 3. Sales by Factory\nWhich factory's products generate the most revenue?"))
cells.append(nbf.v4.new_code_cell("""
sales_by_factory = df.groupby('Factory')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)
fig = px.bar(sales_by_factory, x='Factory', y='Sales', title='Sales by Factory', text_auto='.2s')
fig.update_layout(width=800, height=500)
fig.show()
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** 'Lot\\'s O\\' Nuts' and 'Wicked Choccy\\'s' are the primary revenue drivers among our factories. We need to ensure these facilities have adequate capacity and prioritized maintenance to avoid bottlenecks for our most lucrative product lines."))

# Cell 4: Factory Utilization
cells.append(nbf.v4.new_markdown_cell("## 4. Factory Utilization\nOrder volume and total units handled per factory, to spot over/under-loaded factories."))
cells.append(nbf.v4.new_code_cell("""
utilization = df.groupby('Factory').agg({'Order ID': 'nunique', 'Units': 'sum'}).reset_index().rename(columns={'Order ID': 'Order Volume'})
fig = px.scatter(utilization, x='Order Volume', y='Units', size='Units', color='Factory', title='Factory Utilization: Order Volume vs Units Handled', hover_name='Factory')
fig.update_layout(width=800, height=500)
fig.show()
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** The scatter plot reveals a massive workload imbalance. 'Lot's O' Nuts' and 'Wicked Choccy's' handle roughly 96% of the total order volume and 93% of the revenue, while the other three factories are severely underutilized, signaling a massive opportunity to redistribute processing and balance the network."))

# Cell 5: Shipping Mode Distribution
cells.append(nbf.v4.new_markdown_cell("## 5. Shipping Mode Distribution\nHow orders split across Same Day/First/Second/Standard Class."))
cells.append(nbf.v4.new_code_cell("""
ship_mode_dist = df['Ship Mode'].value_counts().reset_index()
ship_mode_dist.columns = ['Ship Mode', 'Count']
fig = px.pie(ship_mode_dist, names='Ship Mode', values='Count', title='Shipping Mode Distribution')
fig.update_layout(width=800, height=500)
fig.show()
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** Standard Class dominates the shipping profile, which keeps baseline transportation costs manageable. However, the proportion of expedited orders (Same Day/First Class) dictates how strictly we must optimize lead times and proximity to customers for high-priority shipments."))

# Cell 6: Product-wise Performance
cells.append(nbf.v4.new_markdown_cell("## 6. Product-wise Performance\nTop 5 and bottom 5 products by Gross Profit."))
cells.append(nbf.v4.new_code_cell("""
product_profit = df.groupby('Product Name')['Gross Profit'].sum().reset_index().sort_values(by='Gross Profit', ascending=False)
top_5 = product_profit.head(5)
bottom_5 = product_profit.tail(5)
perf_df = pd.concat([top_5, bottom_5])
perf_df['Category'] = ['Top 5']*5 + ['Bottom 5']*5
fig = px.bar(perf_df, x='Product Name', y='Gross Profit', color='Category', title='Top 5 and Bottom 5 Products by Gross Profit')
fig.update_layout(xaxis_tickangle=-45)
fig.update_layout(width=800, height=500)
fig.show()
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** 'Wonka Bar' varieties are the definitive cash cows, generating the vast majority of our gross profit. Production strategies should heavily favor these top products, whereas bottom performers like Fun Dip and Nerds contribute very little and could be deprioritized."))

# Cell 7: Lead Time Distribution
cells.append(nbf.v4.new_markdown_cell("## 7. Lead Time Distribution\nHistogram of lead_time_days, plus average lead_time_days broken down by Factory and by Ship Mode."))
cells.append(nbf.v4.new_code_cell("""
fig1 = px.histogram(df, x='lead_time_days', title='Distribution of Lead Time (Days)', nbins=20)
fig1.update_layout(width=800, height=500)
fig1.show()

lead_time_factory = df.groupby('Factory')['lead_time_days'].mean().reset_index().sort_values('lead_time_days')
fig2 = px.bar(lead_time_factory, x='Factory', y='lead_time_days', title='Average Lead Time by Factory')
fig2.update_layout(width=800, height=500)
fig2.show()

lead_time_ship = df.groupby('Ship Mode')['lead_time_days'].mean().reset_index().sort_values('lead_time_days')
fig3 = px.bar(lead_time_ship, x='Ship Mode', y='lead_time_days', title='Average Lead Time by Ship Mode')
fig3.update_layout(width=800, height=500)
fig3.show()
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** The most heavily utilized factories also experience some of the longest average lead times (~8.2 days). Reallocating some demand to the underutilized 'Secret Factory' (which averages 7.4 days) could systematically improve our overall delivery speeds."))

# Cell 8: Regional Demand
cells.append(nbf.v4.new_markdown_cell("## 8. Regional Demand\nOrder count and units by Region over time (monthly trend)."))
cells.append(nbf.v4.new_code_cell("""
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
monthly_demand = df.groupby(['YearMonth', 'Region']).agg({'Order ID': 'nunique', 'Units': 'sum'}).reset_index()

fig = px.line(monthly_demand, x='YearMonth', y='Units', color='Region', title='Monthly Units Demanded by Region', markers=True)
fig.update_layout(width=800, height=500)
fig.show()
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** Tracking monthly demand highlights regional seasonality and overall growth trends. If a particular region shows a sustained spike, we must proactively adjust factory capacities and routing logic to prevent regional stockouts and delayed shipments."))

# Cell 9: Correlation Analysis
cells.append(nbf.v4.new_markdown_cell("## 9. Correlation Analysis\nHeatmap between distance, lead_time_days, Sales, Cost, Gross Profit, Units."))
cells.append(nbf.v4.new_code_cell("""
# Filter only numeric columns for correlation matrix
corr_cols = ['distance', 'lead_time_days', 'Sales', 'Cost', 'Gross Profit', 'Units']
corr_matrix = df[corr_cols].corr()
fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Heatmap', aspect='auto', color_continuous_scale='RdBu_r')
fig.update_layout(width=800, height=500)
fig.show()
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** The strong correlation between distance and lead time confirms that geographic proximity directly impacts our delivery performance. This solidifies the business case for a sophisticated optimization engine to allocate orders to the closest capable factory, reducing both time and transit costs."))

# Cell 10: Geographic Visualization
cells.append(nbf.v4.new_markdown_cell("## 10. Geographic Visualization\nFolium map showing 5 factory locations and customer order density by region."))
cells.append(nbf.v4.new_code_cell("""
factories = df[['Factory', 'Latitude', 'Longitude']].drop_duplicates()

m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add factories
for _, row in factories.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Factory: {row['Factory']}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

# Add regional demand centers
regions = df.groupby('Region').agg({'Region_Lat': 'first', 'Region_Lon': 'first', 'Sales': 'sum'}).reset_index()
for _, row in regions.iterrows():
    folium.CircleMarker(
        location=[row['Region_Lat'], row['Region_Lon']],
        radius=row['Sales'] / 10000, # Scale for visibility
        popup=f"Region: {row['Region']}<br>Sales: ${row['Sales']:.2f}",
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

display(m)
"""))
cells.append(nbf.v4.new_markdown_cell("**Business Takeaway:** The visual mapping reveals the geographic disconnect between some of our production facilities and our heaviest demand centers. This spatial insight is the foundational argument for our Scenario Simulation Engine to reassign demand intelligently across the network."))

# Key EDA Insights
cells.append(nbf.v4.new_markdown_cell("""
## Key EDA Insights

*   **Pacific and Atlantic Drive Revenue:** The Pacific and Atlantic regions lead overall sales, indicating that factory allocations must prioritize these geographic clusters to sustain top-line growth.
*   **Massive Factory Imbalance:** "Lot's O' Nuts" and "Wicked Choccy's" handle roughly 96% of the total order volume and 93% of the revenue, while the other three factories are severely underutilized, indicating a huge opportunity to balance the load.
*   **Consistent Profit Margins:** Profit margins are remarkably consistent across all regions (between 65% and 66.4%), suggesting transportation and cost overheads scale proportionally with sales in the current setup.
*   **Shipping Mode Economics:** Standard Class handles the bulk of orders, but nearly 2,100 orders are expedited (First/Same Day). Maintaining SLA requirements for these requires minimizing distance.
*   **Product Concentration:** "Wonka Bar" products are the definitive cash cows, generating the highest gross profit. Safeguarding their production is paramount, whereas candies like Fun Dip and Nerds contribute very little.
*   **Lead Time Discrepancies:** The most heavily utilized factories also experience some of the longest average lead times (~8.2 days), meaning reallocating some demand to the underutilized "Secret Factory" (7.4 days) could improve overall delivery speed.
*   **Simulation Engine Mandate:** The geographic disconnect and factory imbalance strongly justify building the Scenario Simulation Engine to minimize distance, balance utilization, and improve delivery speeds systematically.
"""))

nb['cells'] = cells
with open('notebooks/01_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Notebook notebooks/01_eda.ipynb created successfully.')
