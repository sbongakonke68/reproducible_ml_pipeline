# train_enhanced.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import our preprocessing function
from preprocess_data import preprocess_data

def evaluate_model(model, X_test, y_test, model_name):
    """
    Comprehensive evaluation of a trained model.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    # Calculate percentage error (MAPE alternative for regression)
    percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"- Mean Squared Error (MSE): {mse:.4f}")
    print(f"- Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"- Mean Absolute Error (MAE): {mae:.4f}")
    print(f"- R² Score: {r2:.4f}")
    print(f"- Mean Percentage Error: {percentage_error:.2f}%")
    
    return {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'percentage_error': percentage_error,
        'predictions': y_pred
    }

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models.
    """
    try:
        # Get feature importance
        importances = model.feature_importances_
        
        # Create a DataFrame for easier plotting
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_imp_df.head(10), x='importance', y='feature')
        plt.title(f'Top 10 Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'model/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved: model/{model_name}_feature_importance.png")
        
    except Exception as e:
        print(f"Could not plot feature importance for {model_name}: {e}")

def plot_predictions_vs_actual(y_test, y_pred, model_name):
    """
    Plot predicted vs actual values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.tight_layout()
    plt.savefig(f'model/{model_name}_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Predictions vs actual plot saved: model/{model_name}_predictions_vs_actual.png")

def main():
    print("Starting enhanced model training pipeline...")
    
    # 1. Load the raw data
    df = pd.read_csv('data/raw/winequality-red.csv')
    print("Raw data loaded successfully.")
    print(f"Original data shape: {df.shape}")
    
    # 2. Preprocess the data
    X_train, X_test, y_train, y_test, training_params = preprocess_data(df, is_training=True)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # 3. Define multiple models to try
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all available cores
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Support Vector Machine': SVR(kernel='rbf', C=1.0)
    }
    
    # 4. Train and evaluate each model
    results = []
    best_model = None
    best_r2 = -np.inf
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        print(f"✅ {model_name} training completed.")
        
        # Evaluate the model
        result = evaluate_model(model, X_test, y_test, model_name)
        results.append(result)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"- Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Plot predictions vs actual for the first model or best model
        if model_name == 'Random Forest' or result['r2'] > best_r2:
            plot_predictions_vs_actual(y_test, result['predictions'], model_name)
        
        # Plot feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, X_train.columns.tolist(), model_name)
        
        # Update best model
        if result['r2'] > best_r2:
            best_r2 = result['r2']
            best_model = model
            best_model_name = model_name
    
    # 5. Compare all models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print('='*60)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2', ascending=False)
    
    print(results_df[['model_name', 'r2', 'rmse', 'mae']].to_string(index=False))
    
    # 6. Save the best model and training parameters
    import os
    os.makedirs('model', exist_ok=True)
    
    # Save best model
    best_model_path = f'model/best_model_{best_model_name.replace(" ", "_").lower()}.joblib'
    joblib.dump(best_model, best_model_path)
    print(f"\n✅ Best model ({best_model_name}) saved to: {best_model_path}")
    
    # Save training parameters (crucial for inference)
    params_path = 'model/training_params.joblib'
    joblib.dump(training_params, params_path)
    print(f"✅ Training parameters saved to: {params_path}")
    
    # Save model comparison results
    results_df.to_csv('model/model_comparison_results.csv', index=False)
    print(f"✅ Model comparison results saved to: model/model_comparison_results.csv")
    
    # 7. Final best model analysis
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"📊 Best R² Score: {best_r2:.4f}")
    print('='*60)
    
    # Feature importance for the best model (if it's tree-based)
    if hasattr(best_model, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_imp.head(10).to_string(index=False))
    
    print("\n🎯 Training pipeline completed successfully!")

if __name__ == "__main__":
    main()