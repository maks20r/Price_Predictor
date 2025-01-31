import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('preprocessed_real_estate.csv')

# Filter for 2024 data and non-null landmark locations
df_filtered = df[df['nearest_landmark_en'] != 'Unknown'].copy()

# Convert price per sqm to price per sq ft for easier understanding
df_filtered['price_per_sqft'] = df_filtered['price_per_sqm'] / 10.764

# Calculate statistics for each landmark
landmark_stats = df_filtered.groupby('nearest_landmark_en').agg({
    'price_per_sqft': ['count', 'mean', 'median', 'std'],
    'actual_worth': ['mean'],
    'property_type_en': lambda x: x.value_counts().index[0]  # Most common property type
}).round(2)

landmark_stats.columns = [
    'number_of_properties', 
    'avg_price_per_sqft', 
    'median_price_per_sqft',
    'std_price_per_sqft',
    'avg_actual_worth',
    'most_common_property_type'
]

# Sort by average price per square foot
landmark_stats_sorted = landmark_stats.sort_values('avg_price_per_sqft', ascending=False)

# Create visualizations
plt.figure(figsize=(15, 8))
plt.bar(landmark_stats_sorted.index, landmark_stats_sorted['avg_price_per_sqft'])
plt.xticks(rotation=45, ha='right')
plt.title('Average Price per Square Foot by Landmark')
plt.xlabel('Landmark')
plt.ylabel('Price per Square Foot ($)')
plt.tight_layout()
plt.savefig('landmark_price_analysis.png')
plt.close()

# Create box plot for price distribution
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_filtered, x='nearest_landmark_en', y='price_per_sqft')
plt.xticks(rotation=45, ha='right')
plt.title('Price Distribution by Landmark')
plt.xlabel('Landmark')
plt.ylabel('Price per Square Foot ($)')
plt.tight_layout()
plt.savefig('landmark_price_distribution.png')
plt.close()

# Prepare the output
output = "Landmark Location Price Analysis\n"
output += "=" * 40 + "\n\n"

# Overall statistics
output += "Overall Statistics:\n"
output += "-" * 20 + "\n"
output += f"Total number of properties analyzed: {len(df_filtered)}\n"
output += f"Number of unique landmarks: {df_filtered['nearest_landmark_en'].nunique()}\n"
output += f"Average price per square foot: ${df_filtered['price_per_sqft'].mean():.2f}\n"
output += f"Median price per square foot: ${df_filtered['price_per_sqft'].median():.2f}\n\n"

# Landmark statistics
output += "Price Analysis by Landmark:\n"
output += "-" * 30 + "\n"

for landmark in landmark_stats_sorted.index:
    output += f"\nLandmark: {landmark}\n"
    output += f"Number of properties: {landmark_stats_sorted.loc[landmark, 'number_of_properties']}\n"
    output += f"Average price per sq ft: ${landmark_stats_sorted.loc[landmark, 'avg_price_per_sqft']:.2f}\n"
    output += f"Median price per sq ft: ${landmark_stats_sorted.loc[landmark, 'median_price_per_sqft']:.2f}\n"
    output += f"Average property value: ${landmark_stats_sorted.loc[landmark, 'avg_actual_worth']:,.2f}\n"
    output += f"Most common property type: {landmark_stats_sorted.loc[landmark, 'most_common_property_type']}\n"

# Add insights
output += "\nKey Insights:\n"
output += "-" * 15 + "\n"
output += f"1. Most expensive landmark area: {landmark_stats_sorted.index[0]} "
output += f"(${landmark_stats_sorted['avg_price_per_sqft'].iloc[0]:.2f}/sq ft)\n"
output += f"2. Least expensive landmark area: {landmark_stats_sorted.index[-1]} "
output += f"(${landmark_stats_sorted['avg_price_per_sqft'].iloc[-1]:.2f}/sq ft)\n"
output += f"3. Price difference between highest and lowest: "
output += f"${(landmark_stats_sorted['avg_price_per_sqft'].iloc[0] - landmark_stats_sorted['avg_price_per_sqft'].iloc[-1]):.2f}/sq ft\n"

# Property type distribution
output += "\nProperty Type Distribution by Landmark:\n"
output += "-" * 35 + "\n"
for landmark in landmark_stats_sorted.index:
    property_types = df_filtered[df_filtered['nearest_landmark_en'] == landmark]['property_type_en'].value_counts()
    output += f"\n{landmark}:\n"
    for prop_type, count in property_types.items():
        output += f"- {prop_type}: {count} properties\n"

# Write to file
with open('landmark_analysis.txt', 'w') as f:
    f.write(output)