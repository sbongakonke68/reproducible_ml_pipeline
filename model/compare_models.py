# compare_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_comparison_report():
    """Create a comprehensive model comparison report"""
    
    # Load the model comparison results from training
    try:
        results_df = pd.read_csv('model/model_comparison_results.csv')
        
        # Create comparison visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: R² Comparison
        plt.subplot(2, 2, 1)
        bars = plt.bar(results_df['model_name'], results_df['r2'], color='lightblue')
        plt.ylabel('R² Score')
        plt.title('Model Comparison: R² Scores')
        plt.xticks(rotation=45)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 2: RMSE Comparison
        plt.subplot(2, 2, 2)
        bars = plt.bar(results_df['model_name'], results_df['rmse'], color='lightcoral')
        plt.ylabel('RMSE')
        plt.title('Model Comparison: RMSE')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 3: MAE Comparison
        plt.subplot(2, 2, 3)
        bars = plt.bar(results_df['model_name'], results_df['mae'], color='lightgreen')
        plt.ylabel('MAE')
        plt.title('Model Comparison: MAE')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 4: Metric Radar Chart (simplified)
        plt.subplot(2, 2, 4)
        metrics = ['r2', 'rmse', 'mae']
        normalized_metrics = results_df.copy()
        
        # Normalize for radar chart (invert RMSE and MAE since lower is better)
        normalized_metrics['r2_norm'] = normalized_metrics['r2']
        normalized_metrics['rmse_norm'] = 1 - (normalized_metrics['rmse'] / normalized_metrics['rmse'].max())
        normalized_metrics['mae_norm'] = 1 - (normalized_metrics['mae'] / normalized_metrics['mae'].max())
        
        # Simple bar comparison of normalized metrics
        x = np.arange(len(results_df))
        width = 0.25
        
        plt.bar(x - width, normalized_metrics['r2_norm'], width, label='R²', alpha=0.8)
        plt.bar(x, normalized_metrics['rmse_norm'], width, label='1-RMSE (norm)', alpha=0.8)
        plt.bar(x + width, normalized_metrics['mae_norm'], width, label='1-MAE (norm)', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Normalized Scores')
        plt.title('Normalized Metric Comparison')
        plt.xticks(x, results_df['model_name'], rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('model/model_comparison_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a detailed comparison table
        comparison_table = results_df.copy()
        comparison_table['Performance_Rank'] = comparison_table['r2'].rank(ascending=False)
        
        print("\n🏆 FINAL MODEL COMPARISON REPORT")
        print("="*60)
        print(comparison_table.to_string(index=False))
        
        # Save detailed report
        comparison_table.to_csv('model/detailed_model_comparison.csv', index=False)
        print(f"\n✅ Model comparison report saved: model/detailed_model_comparison.csv")
        
        # Best model recommendation
        best_model = comparison_table.loc[comparison_table['r2'].idxmax()]
        print(f"\n🎯 RECOMMENDED MODEL: {best_model['model_name']}")
        print(f"   R² Score: {best_model['r2']:.4f}")
        print(f"   RMSE: {best_model['rmse']:.4f}")
        print(f"   MAE: {best_model['mae']:.4f}")
        
    except FileNotFoundError:
        print("❌ Model comparison results not found. Please run train_enhanced.py first.")

if __name__ == "__main__":
    create_comparison_report()