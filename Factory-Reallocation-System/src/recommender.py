import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from the current directory if run as a script
import simulator

def recommend_factory(product_name, priority="balanced", target_region=None):
    # Run the simulation for the product
    sim_results = simulator.simulate_product(product_name)
    
    if target_region:
        sim_results = sim_results[sim_results['Region'] == target_region]
        if sim_results.empty:
            raise ValueError(f"Product '{product_name}' does not ship to region '{target_region}'")
            
    total_regions = sim_results['Region'].nunique()
    current_factory = sim_results[sim_results['Is Current']]['Factory'].iloc[0]
    
    # Aggregate results per factory
    # Calculate: average predicted lead time, average % change vs current, count of High Logistics Risk
    
    # Group by factory
    grouped = sim_results.groupby(['Factory', 'Is Current']).agg(
        avg_lead_time=('Predicted Lead Time', 'mean'),
        avg_pct_change=('Lead Time Change vs Current (%)', 'mean'),
        std_pct_change=('Lead Time Change vs Current (%)', 'std'),
        risk_count=('Risk Flag', lambda x: (x == 'High Logistics Risk').sum())
    ).reset_index()
    
    # Fill NaN std dev (happens if only 1 region) with 0
    grouped['std_pct_change'] = grouped['std_pct_change'].fillna(0)
    
    # Confidence Score: 100 - standard deviation (clamped between 0 and 100)
    grouped['Confidence Score (%)'] = np.clip(100 - grouped['std_pct_change'], 0, 100).round(1)
    
    # Calculate Composite Score
    # speed_score = -avg_pct_change * 5 (to scale it closer to 0-100 range, a 20% improvement = 100)
    grouped['speed_score'] = -grouped['avg_pct_change'] * 5
    
    # risk_score = -(risk_count / total_regions) * 100 (0 is perfect, -100 is terrible)
    grouped['risk_score'] = -(grouped['risk_count'] / total_regions) * 100
    
    if priority == "speed":
        w_speed, w_risk = 0.9, 0.1
    elif priority == "profit":
        w_speed, w_risk = 0.1, 0.9
    else: # balanced
        w_speed, w_risk = 0.5, 0.5
        
    grouped['Composite Score'] = (grouped['speed_score'] * w_speed) + (grouped['risk_score'] * w_risk)
    
    # Rank factories by Composite Score descending
    ranked = grouped.sort_values('Composite Score', ascending=False).reset_index(drop=True)
    
    # Generate recommendation string
    best_option = ranked.iloc[0]
    is_current_best = best_option['Is Current']
    recommended_factory = best_option['Factory']
    
    result_strings = []
    
    if is_current_best:
        header = f"Current Factory: {current_factory}\nRecommended Factory: Keep Current Factory ({current_factory})"
    else:
        header = f"Current Factory: {current_factory}\nRecommended Factory: {recommended_factory}"
        
    result_strings.append(header)
    
    # Helper to generate reason
    def generate_reason(row):
        avg_pct = row['avg_pct_change']
        if avg_pct < 0:
            lt_str = f"Lead time reduced by {abs(avg_pct):.1f}%"
        elif avg_pct > 0:
            lt_str = f"Lead time increased by {avg_pct:.1f}%"
        else:
            lt_str = "No change in average lead time"
            
        risk_c = row['risk_count']
        if risk_c == 0:
            profit_str = "Low operational risk (0 regions flagged)"
        else:
            profit_str = f"{risk_c} of {total_regions} regions flagged high logistics risk"
            
        conf = row['Confidence Score (%)']
        
        return f"Reason:\n- {lt_str}\n- Profit impact: {profit_str}\n- Confidence Score: {conf}%"
    
    result_strings.append(generate_reason(best_option))
    
    result_strings.append("\n--- All Ranked Alternatives ---")
    for i, row in ranked.iterrows():
        rank_str = f"{i+1}. {row['Factory']}"
        if row['Is Current']:
            rank_str += " (Current)"
        
        score = row['Composite Score']
        rank_str += f" | Score: {score:.1f}"
        
        reason = generate_reason(row)
        # Indent the reason string for alternatives
        reason_indented = "\n".join(["    " + line for line in reason.split("\n")])
        
        result_strings.append(rank_str)
        result_strings.append(reason_indented)
        
    final_output = "\n".join(result_strings)
    
    return {
        'top_recommendation': recommended_factory,
        'is_current_best': is_current_best,
        'ranked_df': ranked,
        'report_text': final_output
    }

def main():
    print("Initializing Recommendation Engine...\n")
    simulator.init_simulator()
    
    # Identify sample products
    divisions = simulator._df['Division'].unique()
    sample_products = []
    for div in ['Chocolate', 'Sugar', 'Other']:
        if div in simulator._df['Division'].values:
            prod = simulator._df[simulator._df['Division'] == div]['Product Name'].iloc[0]
            sample_products.append(prod)
            
    if not sample_products:
        sample_products = simulator._df['Product Name'].drop_duplicates().sample(3, random_state=42).tolist()
        
    for i, prod in enumerate(sample_products):
        print("="*80)
        
        if i == 2: # For the 3rd product, demonstrate the priority shift
            print(f"RECOMMENDATIONS FOR PRODUCT: {prod} [Priority: SPEED vs PROFIT]")
            print("="*80)
            
            print(">>> PRIORITY: SPEED")
            res_speed = recommend_factory(prod, priority="speed")
            print(res_speed['report_text'])
            print("\n" + "-"*40 + "\n")
            
            print(">>> PRIORITY: PROFIT")
            res_profit = recommend_factory(prod, priority="profit")
            print(res_profit['report_text'])
            
        else:
            print(f"RECOMMENDATIONS FOR PRODUCT: {prod} [Priority: BALANCED]")
            print("="*80)
            res_balanced = recommend_factory(prod, priority="balanced")
            print(res_balanced['report_text'])
            
        print("\n\n")

if __name__ == "__main__":
    main()
