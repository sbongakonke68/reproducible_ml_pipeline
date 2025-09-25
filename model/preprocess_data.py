import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

def select_features(X, y, is_training=True, feature_selector=None, feature_mask=None):
    """
    Selects relevant features using variance, correlation, and feature importance.
    Avoids data leakage by using training data to fit selectors.
    """
    X_current = X.copy()
    print("   Starting feature selection...")
    
    if is_training:
        # --- Strategy 1: Remove Low Variance Features ---
        selector = VarianceThreshold(threshold=0.01)
        X_high_variance = selector.fit_transform(X_current)
        kept_mask = selector.get_support()
        selected_features = X_current.columns[kept_mask].tolist()
        X_current = X_current[selected_features]
        print(f"   After low variance removal: {X_current.shape[1]} features remaining.")
        
        # --- Strategy 2: Correlation with Target ---
        corr_with_target = X_current.corrwith(y).abs().sort_values(ascending=False)
        corr_threshold = 0.05
        low_corr_features = corr_with_target[corr_with_target < corr_threshold].index
        X_current.drop(columns=low_corr_features, inplace=True, errors='ignore')
        print(f"   After low correlation removal: {X_current.shape[1]} features remaining.")
        if len(low_corr_features) > 0:
            print(f"   Dropped due to low correlation: {list(low_corr_features)}")
        
        # --- Strategy 3: Model-based Importance ---
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_current, y)
        importances = model.feature_importances_
        importance_threshold = 0.01
        importance_mask = importances > importance_threshold
        selected_features_final = X_current.columns[importance_mask].tolist()
        X_selected = X_current[selected_features_final]
        
        print(f"   After model-based selection: {X_selected.shape[1]} features remaining.")
        dropped_final = set(X_current.columns) - set(selected_features_final)
        if dropped_final:
            print(f"   Dropped due to low importance: {list(dropped_final)}")
        
        feature_mask = selected_features_final
        return X_selected, selector, feature_mask
    
    else:
        # --- Inference Path ---
        if feature_selector is not None:
            X_high_variance = feature_selector.transform(X_current)
            kept_mask = feature_selector.get_support()
            selected_features = X_current.columns[kept_mask].tolist()
            X_current = X_current[selected_features]
        
        X_selected = X_current[feature_mask]
        print(f"   Applied feature selection mask from training. {X_selected.shape[1]} features remaining.")
        return X_selected, feature_selector, feature_mask

def extract_features(X, is_training=True, extraction_params=None):
    """
    Simple feature extraction for wine data.
    """
    X_extracted = X.copy()
    print("   Extracting new features...")
    
    try:
        # Simple feature: alcohol to acidity ratio
        if 'alcohol' in X.columns and 'fixed acidity' in X.columns:
            X_extracted['alcohol_acidity_ratio'] = X_extracted['alcohol'] / (X_extracted['fixed acidity'] + 0.001)
            print("     - Added alcohol_acidity_ratio")
        
        # Simple feature: total sulfur impact
        if 'total sulfur dioxide' in X.columns:
            X_extracted['sulfur_impact'] = X_extracted['total sulfur dioxide'] * 0.1
            print("     - Added sulfur_impact")
            
        print(f"     Added {X_extracted.shape[1] - X.shape[1]} new features.")
        return X_extracted, {'features_extracted': True}
        
    except Exception as e:
        print(f"     Error in feature extraction: {e}")
        print("     Returning original features.")
        return X, {'features_extracted': False}

def preprocess_data(df, test_size=0.2, random_state=42, is_training=True, training_params=None):
    """
    A robust function to clean the wine data, avoiding data leakage.
    Now includes feature scaling.
    """
    data = df.copy()
    
    # --- 1. Handle Missing Values ---
    print("1. Handling missing values...")
    if is_training:
        missing_values = data.isnull().sum()
        if missing_values.any():
            print(f"   Found missing values:\n{missing_values[missing_values > 0]}")
            medians = data[missing_values[missing_values > 0].index].median().to_dict()
            data.fillna(medians, inplace=True)
            print("   Filled missing values with medians.")
            training_params = {'medians': medians}
        else:
            print("   No missing values found.")
            training_params = {'medians': {}}
    else:
        if training_params and 'medians' in training_params:
            data.fillna(training_params['medians'], inplace=True)
            print("   Filled missing values using medians from training.")

    # --- 2. Handle Duplicates ---
    print("2. Handling duplicates...")
    initial_shape = data.shape
    data = data.drop_duplicates()
    print(f"   Dropped {initial_shape[0] - data.shape[0]} duplicate rows.")

    # --- 3. Separate Features (X) and Target (y) ---
    if 'quality' in data.columns:
        y = data['quality']
        X = data.drop('quality', axis=1)
    else:
        X = data
        y = None

    # --- 4. Outlier Detection & Treatment (IQR Method) ---
    print("3. Handling outliers...")
    if is_training:
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        training_params['iqr_bounds'] = {'lower': lower_bound, 'upper': upper_bound}
        for col in X.columns:
            X[col] = np.where(X[col] < lower_bound[col], lower_bound[col], X[col])
            X[col] = np.where(X[col] > upper_bound[col], upper_bound[col], X[col])
        print("   Capped outliers using IQR method.")
    else:
        if training_params and 'iqr_bounds' in training_params:
            lb = training_params['iqr_bounds']['lower']
            ub = training_params['iqr_bounds']['upper']
            for col in X.columns:
                if col in lb and col in ub:
                    X[col] = np.where(X[col] < lb[col], lb[col], X[col])
                    X[col] = np.where(X[col] > ub[col], ub[col], X[col])
            print("   Capped outliers using IQR bounds from training.")

    # --- 5. Feature Extraction ---
    print("4. Creating new features through feature extraction...")
    X, extraction_params = extract_features(X, is_training=is_training)
    if is_training:
        training_params['extraction_params'] = extraction_params

    # --- 6. Remove Highly Correlated Features ---
    print("5. Removing highly correlated features...")
    if is_training:
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
        training_params['features_to_drop'] = to_drop
        if to_drop:
            print(f"   Identified redundant features to drop: {to_drop}")
            X.drop(columns=to_drop, inplace=True, errors='ignore')
        else:
            print("   No highly correlated features found.")
    else:
        if training_params and 'features_to_drop' in training_params:
            to_drop = training_params['features_to_drop']
            if to_drop:
                X.drop(columns=to_drop, inplace=True, errors='ignore')
                print(f"   Dropped features based on training list: {to_drop}")

    # --- 7. Feature Selection ---
    print("6. Selecting most important features...")
    if is_training:
        X_processed, variance_selector, feature_mask = select_features(X, y, is_training=True)
        training_params['variance_selector'] = variance_selector
        training_params['feature_mask'] = feature_mask
    else:
        X_processed, _, _ = select_features(X, y, is_training=False, 
                                          feature_selector=training_params.get('variance_selector'),
                                          feature_mask=training_params.get('feature_mask'))

    # --- 8. Feature Scaling (Standardization) --- 
    print("7. Standardizing features (StandardScaler)...")
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        training_params['scaler'] = scaler
        print("   Fitted StandardScaler on training data.")
    else:
        if training_params and 'scaler' in training_params:
            scaler = training_params['scaler']
            X_scaled = scaler.transform(X_processed)
            print("   Scaled features using StandardScaler from training.")
        else:
            X_scaled = X_processed.values
            print("   Warning: No scaler found in training_params. Features not scaled.")

    feature_names = X_processed.columns.tolist()
    X_processed = pd.DataFrame(X_scaled, columns=feature_names)

    # --- 9. Train-Test Split (Only for training) ---
    if is_training and y is not None:
        print("8. Performing train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("   Preprocessing complete.")
        print(f"   Final training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test, training_params
    else:
        print("   Preprocessing for inference complete.")
        return X_processed, training_params

# For testing the script directly
if __name__ == "__main__":
    data_path = 'data/raw/winequality-red.csv'
    df = pd.read_csv(data_path)
    print("Data loaded for preprocessing test.\n")
    X_train, X_test, y_train, y_test, params = preprocess_data(df, is_training=True)
    print("\nPreview of SCALED training features:")
    print(X_train.head())
    print(f"\nScaled features mean: {X_train.mean().values}")
    print(f"Scaled features std: {X_train.std().values}")