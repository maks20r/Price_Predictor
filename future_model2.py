import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set seaborn defaults
sns.set_theme()

# Load and prepare data
def prepare_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Convert price to price per square foot
    df['price_per_sqft'] = df['price_per_sqm'] / 10.764
    
    # Population projections
    current_population = 3.25
    target_population = 5.0
    years = df['year'].unique()
    population_projections = {
        year: current_population + (target_population - current_population) * 
        ((year - min(years)) / (max(years) - min(years)))
        for year in years
    }
    
    # Units added per year
    units_per_year = 100000
    
    # Compute yearly aggregates
    yearly_features = df.groupby('year').agg({
        'price_per_sqft': 'mean',
        'area_name_en': 'count'
    }).reset_index()
    
    yearly_features['population_projection'] = yearly_features['year'].map(population_projections)
    yearly_features['housing_supply_factor'] = yearly_features['area_name_en'] * units_per_year / (current_population * 1000)
    
    return df, yearly_features, years

def train_model(yearly_features):
    # Prepare data for prediction
    X = yearly_features[['year', 'population_projection', 'housing_supply_factor']]
    y = yearly_features['price_per_sqft']
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, poly, X_poly, y, X_train, X_test, y_train, y_test

def predict_future_prices(model, poly, years):
    # Predict future years
    future_years = list(range(2025, 2031))
    current_population = 3.25
    target_population = 5.0
    
    future_population_projections = {
        year: current_population + (target_population - current_population) * 
        ((year - min(years)) / (max(years) - min(years)))
        for year in future_years
    }
    
    units_per_year = 100000
    
    future_data = pd.DataFrame({
        'year': future_years,
        'population_projection': [future_population_projections[year] for year in future_years],
        'housing_supply_factor': [units_per_year / (current_population * 1000)] * len(future_years)
    })
    
    future_data_poly = poly.transform(future_data)
    predicted_prices = model.predict(future_data_poly)
    
    results = pd.DataFrame({
        'Year': future_years,
        'Predicted Price per Sq Ft': predicted_prices
    })
    
    return results

def create_visualizations(df, yearly_features, model, X_poly, y, results):
    # 1. Historical vs Predicted Prices Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='year', y='price_per_sqft')
    plt.title('Historical Price Distribution by Year', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price per Square Foot ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = yearly_features[['price_per_sqft', 'population_projection', 'housing_supply_factor']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap', fontsize=15)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # 3. Residual Plot
    y_pred = model.predict(X_poly)
    residuals = y - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot', fontsize=15)
    plt.tight_layout()
    plt.savefig('residual_plot.png')
    plt.close()

    # 4. Price Trend with Confidence Intervals
    plt.figure(figsize=(12, 6))
    
    # Calculate confidence intervals
    confidence = 0.95
    degrees_of_freedom = len(y) - 2
    mse = np.mean((y - y_pred) ** 2)
    standard_error = np.sqrt(mse)
    
    critical_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    margin_of_error = critical_value * standard_error
    
    # Plot historical and predicted data
    historical_years = yearly_features['year']
    historical_prices = yearly_features['price_per_sqft']
    
    plt.plot(historical_years, historical_prices, 'o-', label='Historical Prices')
    plt.plot(results['Year'], results['Predicted Price per Sq Ft'], 'r--', label='Predicted Prices')
    
    plt.fill_between(results['Year'],
                     results['Predicted Price per Sq Ft'] - margin_of_error,
                     results['Predicted Price per Sq Ft'] + margin_of_error,
                     alpha=0.2, color='red', label=f'{int(confidence*100)}% Confidence Interval')
    
    plt.title('Historical and Predicted Prices with Confidence Intervals', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price per Square Foot ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('price_trends_with_ci.png')
    plt.close()

    # 5. Feature Importance Bar Plot
    # Get the feature names from the polynomial features
    poly = PolynomialFeatures(degree=2)
    X = yearly_features[['year', 'population_projection', 'housing_supply_factor']]
    poly.fit_transform(X)  # Fit to get feature names
    feature_names = poly.get_feature_names_out(['year', 'population_projection', 'housing_supply_factor'])
    coefficients = model.coef_
    
    # Ensure lengths match
    if len(feature_names) != len(coefficients):
        feature_names = feature_names[:len(coefficients)]
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(coefficients)), np.abs(coefficients))
    plt.xticks(range(len(coefficients)), feature_names, rotation=45, ha='right')
    plt.title('Feature Importance (Absolute Coefficient Values)', fontsize=15)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Absolute Coefficient Value', fontsize=12)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # 6. Original line plot with uncertainty band
    plt.figure(figsize=(12, 6))
    plt.plot(results['Year'], results['Predicted Price per Sq Ft'], marker='o', color='blue')
    plt.title('Real Estate Price Predictions (2025-2030)', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price per Square Foot ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.fill_between(results['Year'],
                     results['Predicted Price per Sq Ft'] * 0.9,
                     results['Predicted Price per Sq Ft'] * 1.1,
                     alpha=0.2, color='blue')
    plt.tight_layout()
    plt.savefig('future_model.png')
    plt.close()

def save_results(model, X_test, y_test, results):
    # Save results to text file
    with open('model_results.txt', 'w') as f:
        f.write("Real Estate Price Prediction Model Results\n")
        f.write("=" * 45 + "\n\n")
        f.write("Model Performance:\n")
        y_pred = model.predict(X_test)
        f.write(f"R² Score: {r2_score(y_test, y_pred):.4f}\n")
        f.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}\n\n")
        f.write("Predicted Prices per Square Foot:\n")
        f.write(results.to_string(index=False))

def main():
    # Load and prepare data
    df, yearly_features, years = prepare_data('preprocessed_real_estate.csv')
    
    # Train model
    model, poly, X_poly, y, X_train, X_test, y_train, y_test = train_model(yearly_features)
    
    # Make predictions
    results = predict_future_prices(model, poly, years)
    
    # Create visualizations
    create_visualizations(df, yearly_features, model, X_poly, y, results)
    
    # Save results
    save_results(model, X_test, y_test, results)
    
    print("Analysis completed. Results and visualizations have been saved.")

if __name__ == "__main__":
    main()