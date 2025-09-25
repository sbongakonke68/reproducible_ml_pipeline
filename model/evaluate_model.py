# evaluate_model.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           explained_variance_score, mean_absolute_percentage_error)
from sklearn.model_selection import learning_curve, validation_curve
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self, model_path, params_path, X_test, y_test, model_name):
        self.model = joblib.load(model_path)
        self.training_params = joblib.load(params_path)
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = self.model.predict(X_test)
        
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'Mean Squared Error (MSE)': mean_squared_error(self.y_test, self.y_pred),
            'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
            'Mean Absolute Error (MAE)': mean_absolute_error(self.y_test, self.y_pred),
            'R² Score': r2_score(self.y_test, self.y_pred),
            'Explained Variance Score': explained_variance_score(self.y_test, self.y_pred),
            'Mean Absolute Percentage Error (MAPE)': mean_absolute_percentage_error(self.y_test, self.y_pred) * 100,
        }
        
        # Additional custom metrics
        residuals = self.y_test - self.y_pred
        metrics['Mean Residual'] = np.mean(residuals)
        metrics['Std of Residuals'] = np.std(residuals)
        metrics['Max Error'] = np.max(np.abs(residuals))
        
        return metrics
    
    def plot_residual_analysis(self):
        """Create comprehensive residual analysis plots"""
        residuals = self.y_test - self.y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Residual Analysis - {self.model_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Residuals vs Predicted
        axes[0, 0].scatter(self.y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot for normality
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot: Checking Normality of Residuals')
        
        # Plot 3: Distribution of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Actual vs Predicted
        axes[1, 1].scatter(self.y_test, self.y_pred, alpha=0.6)
        max_val = max(self.y_test.max(), self.y_pred.max())
        min_val = min(self.y_test.min(), self.y_pred.min())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'model/{self.model_name}_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Residual analysis plots saved: model/{self.model_name}_residual_analysis.png")
    
    def plot_error_distribution(self):
        """Plot the distribution of errors"""
        errors = self.y_test - self.y_pred
        absolute_errors = np.abs(errors)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Absolute errors distribution
        axes[0].hist(absolute_errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
        axes[0].axvline(absolute_errors.mean(), color='red', linestyle='--', 
                       label=f'Mean: {absolute_errors.mean():.3f}')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Absolute Error Distribution - {self.model_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative error distribution
        sorted_errors = np.sort(absolute_errors)
        cumulative = np.cumsum(sorted_errors) / np.sum(absolute_errors)
        axes[1].plot(sorted_errors, cumulative, linewidth=2)
        axes[1].set_xlabel('Absolute Error')
        axes[1].set_ylabel('Cumulative Proportion')
        axes[1].set_title(f'Cumulative Error Distribution - {self.model_name}')
        axes[1].grid(True, alpha=0.3)
        
        # Add some percentiles
        for percentile in [50, 80, 90, 95]:
            threshold = np.percentile(absolute_errors, percentile)
            axes[1].axvline(threshold, color='red', linestyle='--', alpha=0.7)
            axes[1].text(threshold, 0.1, f'{percentile}%', rotation=90)
        
        plt.tight_layout()
        plt.savefig(f'model/{self.model_name}_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Error distribution plots saved: model/{self.model_name}_error_distribution.png")
    
    def plot_confidence_intervals(self):
        """Plot predictions with confidence intervals"""
        # For tree-based models, we can estimate confidence intervals
        if hasattr(self.model, 'estimators_'):
            # Get predictions from each tree in the forest
            predictions = []
            for tree in self.model.estimators_:
                predictions.append(tree.predict(self.X_test))
            
            predictions = np.array(predictions)
            mean_predictions = np.mean(predictions, axis=0)
            std_predictions = np.std(predictions, axis=0)
            
            plt.figure(figsize=(12, 8))
            
            # Sort by actual values for better visualization
            sorted_idx = np.argsort(self.y_test)
            y_test_sorted = self.y_test.iloc[sorted_idx]
            mean_pred_sorted = mean_predictions[sorted_idx]
            std_pred_sorted = std_predictions[sorted_idx]
            
            plt.plot(y_test_sorted.values, label='Actual', linewidth=2)
            plt.plot(mean_pred_sorted, label='Predicted', linewidth=2)
            plt.fill_between(range(len(y_test_sorted)), 
                           mean_pred_sorted - 1.96 * std_pred_sorted,
                           mean_pred_sorted + 1.96 * std_pred_sorted,
                           alpha=0.3, label='95% Confidence Interval')
            
            plt.xlabel('Sample Index (sorted by actual value)')
            plt.ylabel('Wine Quality')
            plt.title(f'Predictions with Confidence Intervals - {self.model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'model/{self.model_name}_confidence_intervals.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Confidence interval plot saved: model/{self.model_name}_confidence_intervals.png")
    
    def performance_by_quality_level(self):
        """Analyze performance for different wine quality levels"""
        results = []
        quality_levels = sorted(self.y_test.unique())
        
        for quality in quality_levels:
            mask = self.y_test == quality
            if mask.sum() > 0:  # Ensure we have samples for this quality level
                y_true_subset = self.y_test[mask]
                y_pred_subset = self.y_pred[mask]
                
                results.append({
                    'Quality': quality,
                    'Samples': mask.sum(),
                    'MAE': mean_absolute_error(y_true_subset, y_pred_subset),
                    'RMSE': np.sqrt(mean_squared_error(y_true_subset, y_pred_subset)),
                    'Bias': np.mean(y_pred_subset - y_true_subset)  # Positive = overestimation
                })
        
        results_df = pd.DataFrame(results)
        
        # Plot performance by quality level
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance by Wine Quality Level - {self.model_name}', fontsize=16, fontweight='bold')
        
        # MAE by quality
        axes[0, 0].bar(results_df['Quality'].astype(str), results_df['MAE'], color='skyblue')
        axes[0, 0].set_xlabel('Wine Quality')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_title('MAE by Quality Level')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Bias by quality
        colors = ['red' if x > 0 else 'green' for x in results_df['Bias']]
        axes[0, 1].bar(results_df['Quality'].astype(str), results_df['Bias'], color=colors)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].set_xlabel('Wine Quality')
        axes[0, 1].set_ylabel('Bias (Predicted - Actual)')
        axes[0, 1].set_title('Bias by Quality Level\n(Red = Overestimation, Green = Underestimation)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sample distribution
        axes[1, 0].bar(results_df['Quality'].astype(str), results_df['Samples'], color='lightgreen')
        axes[1, 0].set_xlabel('Wine Quality')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Sample Distribution by Quality Level')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # RMSE by quality
        axes[1, 1].bar(results_df['Quality'].astype(str), results_df['RMSE'], color='orange')
        axes[1, 1].set_xlabel('Wine Quality')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('RMSE by Quality Level')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'model/{self.model_name}_performance_by_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance by quality level saved: model/{self.model_name}_performance_by_quality.png")
        return results_df
    
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE MODEL EVALUATION REPORT: {self.model_name}")
        print(f"{'='*80}")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        print("\n📈 PERFORMANCE METRICS:")
        print("-" * 40)
        for metric, value in metrics.items():
            if 'Percentage' in metric:
                print(f"{metric:<35}: {value:>8.2f}%")
            else:
                print(f"{metric:<35}: {value:>8.4f}")
        
        # Residual analysis
        print(f"\n🔍 RESIDUAL ANALYSIS:")
        print("-" * 40)
        residuals = self.y_test - self.y_pred
        print(f"Mean of residuals: {residuals.mean():.4f} (should be close to 0)")
        print(f"Standard deviation of residuals: {residuals.std():.4f}")
        print(f"Normality test (p-value): {stats.normaltest(residuals).pvalue:.4f}")
        
        # Performance by quality level
        print(f"\n🍷 PERFORMANCE BY WINE QUALITY LEVEL:")
        print("-" * 40)
        quality_performance = self.performance_by_quality_level()
        print(quality_performance.to_string(index=False))
        
        # Generate all plots
        self.plot_residual_analysis()
        self.plot_error_distribution()
        self.plot_confidence_intervals()
        
        print(f"\n✅ Evaluation complete! Check the 'model' folder for visualization files.")
        
        return metrics, quality_performance

def main():
    # Load the test data
    from preprocess_data import preprocess_data
    
    print("Loading and preprocessing data for evaluation...")
    df = pd.read_csv('data/raw/winequality-red.csv')
    
    # Preprocess the data (this will give us the test set)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, is_training=True)
    
    print(f"Test set shape: {X_test.shape}")
    print(f"Quality distribution in test set:\n{y_test.value_counts().sort_index()}")
    
    # Try to evaluate the best model (adjust the path based on your actual best model)
    model_files = [
        'model/best_model_random_forest.joblib',
        'model/best_model_gradient_boosting.joblib',
        'model/best_model_linear_regression.joblib'
    ]
    
    for model_file in model_files:
        try:
            # Extract model name from filename
            model_name = model_file.split('_')[-1].split('.')[0].title()
            
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} model...")
            print(f"{'='*60}")
            
            evaluator = ModelEvaluator(model_file, 'model/training_params.joblib', 
                                     X_test, y_test, f"{model_name} Model")
            metrics, quality_performance = evaluator.generate_report()
            
            # Save metrics to file
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f'model/{model_name}_detailed_metrics.csv', index=False)
            
        except FileNotFoundError:
            print(f"❌ Model file {model_file} not found. Skipping...")
        except Exception as e:
            print(f"❌ Error evaluating {model_file}: {e}")

if __name__ == "__main__":
    main()