import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the string (in practice, this would be from a file)
df = pd.read_csv('preprocessed_real_estate.csv')

# Convert price_per_sqm to price per square foot (1 square meter = 10.764 square feet)
df['price_per_sqft'] = df['price_per_sqm'] / 10.764

# Group by year and calculate average price per square foot
yearly_avg = df.groupby('year')['price_per_sqft'].mean().round(2)

# Calculate year-over-year percentage change
yearly_pct_change = yearly_avg.pct_change() * 100

# Find the year with highest average price per square foot
max_year = yearly_avg.idxmax()
max_price = yearly_avg.max()

# Create output string
output = "Real Estate Price Analysis\n"
output += "=" * 25 + "\n\n"

output += "Average Price per Square Foot by Year:\n"
output += "-" * 35 + "\n"
for year, price in yearly_avg.items():
    output += f"{year}: ${price:.2f}\n"

output += "\nYear-over-Year Percentage Changes:\n"
output += "-" * 35 + "\n"
for year, change in yearly_pct_change.items():
    if not pd.isna(change):
        output += f"{year}: {change:.1f}%\n"

output += f"\nYear with Highest Average Price: {max_year}"
output += f"\nHighest Average Price: ${max_price:.2f} per sq ft"

# Write to file
with open('final.txt', 'w') as f:
    f.write(output)

# Create the bar plot
plt.figure(figsize=(15, 8))
plt.bar(yearly_avg.index, yearly_avg.values, color='skyblue')
plt.xticks(yearly_avg.index, rotation=45, ha='right')
plt.title('Average Price per Square Foot by Year', pad=20, size=14)
plt.xlabel('Year', size=12)
plt.ylabel('Price per Square Foot ($)', size=12)

# Add value labels on top of each bar
for i, v in enumerate(yearly_avg.values):
    plt.text(yearly_avg.index[i], v, f'${v:.2f}', 
             ha='center', va='bottom')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('yearly_price_analysis.png', dpi=300, bbox_inches='tight')
plt.close()