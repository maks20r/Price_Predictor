import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme()

class ModelEvaluator:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
        }
        self.results = {}
        self.predictions = {}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        evaluation_metrics = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            evaluation_metrics[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_mean': cv_scores.mean(),
                'CV_std': cv_scores.std()
            }
            
            self.results[name] = model
            self.predictions[name] = y_pred
            
        return evaluation_metrics

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

def prepare_features(yearly_features):
    # Prepare data for prediction
    X = yearly_features[['year', 'population_projection', 'housing_supply_factor']]
    y = yearly_features['price_per_sqft']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, poly, scaler, X_poly, y

def predict_future_prices(models, poly, scaler, years):
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
    
    # Scale and transform features
    future_data_scaled = scaler.transform(future_data)
    future_data_poly = poly.transform(future_data_scaled)
    
    # Make predictions with all models
    predictions = {}
    for name, model in models.items():
        predicted_prices = model.predict(future_data_poly)
        predictions[name] = predicted_prices
    
    # Create results dataframe
    results = pd.DataFrame({
        'Year': future_years
    })
    
    for name, preds in predictions.items():
        results[f'{name} Prediction'] = preds
    
    return results

def create_model_comparison_plots(evaluation_metrics, results, yearly_features):
    # 1. Model Performance Comparison
    plt.figure(figsize=(12, 6))
    metrics_df = pd.DataFrame(evaluation_metrics).T
    metrics_df[['R2', 'CV_mean']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend(['R² Score', 'Cross-validation Score'])
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    # 2. Prediction Comparison Plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(yearly_features['year'], yearly_features['price_per_sqft'], 
             'o-', label='Historical Prices', color='black')
    
    # Plot predictions from each model
    colors = ['red', 'blue', 'green']
    for (name, _), color in zip(evaluation_metrics.items(), colors):
        plt.plot(results['Year'], results[f'{name} Prediction'], 
                '--', label=f'{name} Predictions', color=color)
    
    plt.title('Price Predictions by Different Models')
    plt.xlabel('Year')
    plt.ylabel('Price per Square Foot ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_predictions_comparison.png')
    plt.close()

def save_comparison_results(evaluation_metrics, results):
    with open('model_comparison_results.txt', 'w') as f:
        f.write("Real Estate Price Prediction - Model Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        # Write metrics for each model
        f.write("Model Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        for model_name, metrics in evaluation_metrics.items():
            f.write(f"\n{model_name}:\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("\nPredicted Prices per Square Foot:\n")
        f.write(results.to_string(index=False))

def create_visualizations(df, yearly_features, models, X_poly, y, results):

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = yearly_features[['price_per_sqft', 'population_projection', 'housing_supply_factor']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap', fontsize=15)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def main():
    # Load and prepare data
    df, yearly_features, years = prepare_data('preprocessed_real_estate.csv')
    
    # Prepare features
    X_train, X_test, y_train, y_test, poly, scaler, X_poly, y = prepare_features(yearly_features)
    
    # Train and evaluate models
    evaluator = ModelEvaluator()
    evaluation_metrics = evaluator.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Make predictions with all models
    results = predict_future_prices(evaluator.results, poly, scaler, years)
    
    # Create comparison visualizations
    create_model_comparison_plots(evaluation_metrics, results, yearly_features)
    
    # Save comparison results
    save_comparison_results(evaluation_metrics, results)
    
    # Create visualizations
    create_visualizations(df, yearly_features, evaluator.results, X_poly, y, results)
    
    print("Analysis completed. Results and visualizations have been saved.")

if __name__ == "__main__":
    main()