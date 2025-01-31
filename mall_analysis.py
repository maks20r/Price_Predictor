import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('preprocessed_real_estate.csv')

# Filter for non-null mall locations
df_filtered = df[df['nearest_mall_en'] != 'Unknown'].copy()

# Convert price per sqm to price per sq ft for easier understanding
df_filtered['price_per_sqft'] = df_filtered['price_per_sqm'] / 10.764

# Calculate statistics for each mall
mall_stats = df_filtered.groupby('nearest_mall_en').agg({
    'price_per_sqft': ['count', 'mean', 'median', 'std'],
    'actual_worth': ['mean'],
    'property_type_en': lambda x: x.value_counts().index[0],  # Most common property type
    'property_usage_en': lambda x: x.value_counts().index[0]  # Most common usage
}).round(2)

mall_stats.columns = [
    'number_of_properties', 
    'avg_price_per_sqft', 
    'median_price_per_sqft',
    'std_price_per_sqft',
    'avg_actual_worth',
    'most_common_property_type',
    'most_common_usage'
]

# Sort by average price per square foot
mall_stats_sorted = mall_stats.sort_values('avg_price_per_sqft', ascending=False)

# Create visualization for average prices
plt.figure(figsize=(15, 8))
plt.bar(mall_stats_sorted.index, mall_stats_sorted['avg_price_per_sqft'])
plt.xticks(rotation=45, ha='right')
plt.title('Average Price per Square Foot by Nearest Mall')
plt.xlabel('Mall')
plt.ylabel('Price per Square Foot ($)')
plt.tight_layout()
plt.savefig('mall_price_analysis.png')
plt.close()

# Create box plot for price distribution
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_filtered, x='nearest_mall_en', y='price_per_sqft')
plt.xticks(rotation=45, ha='right')
plt.title('Price Distribution by Nearest Mall')
plt.xlabel('Mall')
plt.ylabel('Price per Square Foot ($)')
plt.tight_layout()
plt.savefig('mall_price_distribution.png')
plt.close()

# Prepare the output
output = "Mall Location Price Analysis\n"
output += "=" * 40 + "\n\n"

# Overall statistics
output += "Overall Statistics:\n"
output += "-" * 20 + "\n"
output += f"Total number of properties analyzed: {len(df_filtered)}\n"
output += f"Number of unique malls: {df_filtered['nearest_mall_en'].nunique()}\n"
output += f"Average price per square foot: ${df_filtered['price_per_sqft'].mean():.2f}\n"
output += f"Median price per square foot: ${df_filtered['price_per_sqft'].median():.2f}\n\n"

# Mall statistics
output += "Price Analysis by Mall:\n"
output += "-" * 30 + "\n"

for mall in mall_stats_sorted.index:
    output += f"\nMall: {mall}\n"
    output += f"Number of properties: {mall_stats_sorted.loc[mall, 'number_of_properties']}\n"
    output += f"Average price per sq ft: ${mall_stats_sorted.loc[mall, 'avg_price_per_sqft']:.2f}\n"
    output += f"Median price per sq ft: ${mall_stats_sorted.loc[mall, 'median_price_per_sqft']:.2f}\n"
    output += f"Average property value: ${mall_stats_sorted.loc[mall, 'avg_actual_worth']:,.2f}\n"
    output += f"Most common property type: {mall_stats_sorted.loc[mall, 'most_common_property_type']}\n"
    output += f"Most common usage: {mall_stats_sorted.loc[mall, 'most_common_usage']}\n"

# Add insights
output += "\nKey Insights:\n"
output += "-" * 15 + "\n"
output += f"1. Most expensive mall area: {mall_stats_sorted.index[0]} "
output += f"(${mall_stats_sorted['avg_price_per_sqft'].iloc[0]:.2f}/sq ft)\n"
output += f"2. Least expensive mall area: {mall_stats_sorted.index[-1]} "
output += f"(${mall_stats_sorted['avg_price_per_sqft'].iloc[-1]:.2f}/sq ft)\n"
output += f"3. Price difference between highest and lowest: "
output += f"${(mall_stats_sorted['avg_price_per_sqft'].iloc[0] - mall_stats_sorted['avg_price_per_sqft'].iloc[-1]):.2f}/sq ft\n"

# Property type and usage distribution
output += "\nProperty Distribution by Mall:\n"
output += "-" * 35 + "\n"
for mall in mall_stats_sorted.index:
    mall_properties = df_filtered[df_filtered['nearest_mall_en'] == mall]
    
    output += f"\n{mall}:\n"
    output += "Property Types:\n"
    for prop_type, count in mall_properties['property_type_en'].value_counts().items():
        output += f"- {prop_type}: {count} properties\n"
    
    output += "Property Usage:\n"
    for usage, count in mall_properties['property_usage_en'].value_counts().items():
        output += f"- {usage}: {count} properties\n"

# Additional analysis of price trends
avg_prices_by_year = df_filtered.groupby(['nearest_mall_en', 'year'])['price_per_sqft'].mean().unstack()
output += "\nPrice Trends:\n"
output += "-" * 15 + "\n"
output += str(avg_prices_by_year.round(2))

# Write to file
with open('mall_analysis.txt', 'w') as f:
    f.write(output)