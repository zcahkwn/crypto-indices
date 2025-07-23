from bitcoin_predictor import BitcoinPricePredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Define data paths
data_paths = {
    'bitcoin': 'bitcoin_data.csv',      # Your Bitcoin OHLCV data
    'google': 'google_trends.csv',      # Your Google trends data  
    'fear_greed': 'fear_greed.csv'      # Your Fear & Greed index data
}

# Enhanced model parameters
enhanced_params = {
    'n_estimators': 1500,
    'learning_rate': 0.005,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42
}

print("Initializing Enhanced Bitcoin Price Predictor...")
predictor = BitcoinPricePredictor(
    data_paths=data_paths,
    model_params=enhanced_params
)

try:
    # Execute enhanced pipeline
    print("Starting feature engineering...")
    #predictor.create_features(target_type='multiclass')  # Use multiclass targets
    predictor.create_features_simple(target_type='multiclass')
    # Skip the lag optimization step
    predictor.train_model(n_splits=5)

    # print("\nOptimizing feature lags...")
    # predictor, optimal_lag, lag_results = predictor.optimize_feature_lags()
    
    print("\nOptimizing feature lags...")
    try:
        predictor, optimal_lag, lag_results = predictor.optimize_feature_lags()
        # Ensure lag_results is a dictionary
        if not isinstance(lag_results, dict):
            print(f"Warning: lag_results is not a dict, got {type(lag_results)}")
            lag_results = {}
            optimal_lag = 0
    except Exception as e:
        print(f"Lag optimization failed: {e}")
        lag_results = {}
        optimal_lag = 0
        
    print("\nStarting model training...")
    predictor.train_model(n_splits=5)
    
    # Save results
    predictor.save_model('enhanced_bitcoin_predictor.pkl')
    
    # Generate predictions
    print("\nGenerating sample predictions...")
    recent_data = predictor.feature_df.tail(30)
    predictions, probabilities = predictor.predict(recent_data)
    
    # Enhanced results analysis
    results_df = pd.DataFrame({
        'date': recent_data['date'].values,
        'actual_price': recent_data['close'].values,
        'predicted_class': predictions
    })
    
    # Add probability columns based on number of classes
    n_classes = probabilities.shape[1]
    if n_classes == 4:  # Multiclass: -1, 0, 1, 2
        results_df['prob_strong_down'] = probabilities[:, 0]
        results_df['prob_weak_down'] = probabilities[:, 1]
        results_df['prob_weak_up'] = probabilities[:, 2]
        results_df['prob_strong_up'] = probabilities[:, 3]
    elif n_classes == 2:  # Binary: 0, 1
        results_df['prob_down'] = probabilities[:, 0]
        results_df['prob_up'] = probabilities[:, 1]
    else:  # Handle other cases
        for i in range(n_classes):
            results_df[f'prob_class_{i}'] = probabilities[:, i]
    
    print("\nSample Predictions (Last 30 days):")
    print(results_df.head(10))
    
    # Enhanced visualization
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Price with multiclass predictions
    plt.subplot(3, 2, 1)
    plt.plot(results_df['date'], results_df['actual_price'], label='BTC Price', linewidth=2)
    
    # Color code by prediction class
    class_colors = {0: 'red', 1: 'orange', 2: 'lightgreen', 3: 'green'}
    class_labels = {0: 'Strong Down', 1: 'Weak Down', 2: 'Weak Up', 3: 'Strong Up'}

    
    for class_val, color in class_colors.items():
        mask = results_df['predicted_class'] == class_val
        if mask.any():
            plt.scatter(results_df[mask]['date'], results_df[mask]['actual_price'], 
                        color=color, alpha=0.7, label=class_labels[class_val], s=60)
    
    plt.title('Bitcoin Price with Multiclass Predictions')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Lag optimization results
    # plt.subplot(3, 2, 2)
    # if lag_results:
    #     lags = list(lag_results.keys())
    #     accuracies = list(lag_results.values())
    #     plt.plot(lags, accuracies, marker='o', linewidth=2, markersize=8)
    #     plt.title('Lag Optimization Results')
    #     plt.xlabel('Lag (days)')
    #     plt.ylabel('Cross-validation Accuracy')
    #     plt.grid(True, alpha=0.3)
    #     plt.axvline(x=optimal_lag, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_lag} days')
    #     plt.legend()
    plt.subplot(3, 2, 2)
    if lag_results and isinstance(lag_results, dict):  # Check it's a dict
        lags = list(lag_results.keys())
        accuracies = list(lag_results.values())
        plt.plot(lags, accuracies, marker='o', linewidth=2, markersize=8)
        plt.title('Lag Optimization Results')
        plt.xlabel('Lag (days)')
        plt.ylabel('Cross-validation Accuracy')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=optimal_lag, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_lag} days')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Lag optimization not available', 
                transform=plt.gca().transAxes, ha='center', va='center')
        plt.title('Lag Optimization Results - Not Available')
    
    # Plot 3: Feature importance
    plt.subplot(3, 2, 3)
    feature_importance = pd.DataFrame({
        'feature': predictor.feature_columns,
        'importance': predictor.model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    sns.barplot(data=feature_importance, y='feature', x='importance')
    plt.title('Top 20 Feature Importance')
    
    # Plot 4: Prediction confidence distribution
    plt.subplot(3, 2, 4)
    max_probs = np.max(probabilities, axis=1)
    plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Maximum Probability')
    plt.ylabel('Frequency')
    
    # Plot 5: Market phases with enhanced view
    plt.subplot(3, 2, 5)
    market_data = predictor.feature_df.tail(200)
    bull_periods = market_data[market_data['market_phase'] == 1]
    bear_periods = market_data[market_data['market_phase'] == 0]
    
    plt.plot(market_data['date'], market_data['close'], color='black', alpha=0.7, linewidth=1)
    plt.scatter(bull_periods['date'], bull_periods['close'], color='green', alpha=0.5, s=15, label='Bull Market')
    plt.scatter(bear_periods['date'], bear_periods['close'], color='red', alpha=0.5, s=15, label='Bear Market')
    plt.title('Market Phases (Last 200 days)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 6: Class distribution
    plt.subplot(3, 2, 6)
    class_counts = pd.Series(predictions).value_counts().sort_index()
    colors = ['red', 'orange', 'lightgreen', 'green'][:len(class_counts)]
    plt.bar(class_counts.index, class_counts.values, color=colors)
    plt.title('Predicted Class Distribution')
    plt.xlabel('Prediction Class')
    plt.ylabel('Count')
    
    # Set x-tick labels
    tick_labels = []
    for x in class_counts.index:
        if x in class_labels:
            tick_labels.append(class_labels[x])
        else:
            tick_labels.append(str(x))
    plt.xticks(class_counts.index, tick_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('enhanced_bitcoin_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save enhanced results
    results_df.to_csv('enhanced_prediction_results.csv', index=False)
    
    # Save lag optimization results
    if lag_results:
        lag_df = pd.DataFrame(list(lag_results.items()), columns=['lag_days', 'accuracy'])
        lag_df.to_csv('lag_optimization_results.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"- Enhanced predictions: 'enhanced_prediction_results.csv'")
    if lag_results:
        print(f"- Lag optimization: 'lag_optimization_results.csv'")
    print(f"- Visualization: 'enhanced_bitcoin_prediction_results.png'")
    print(f"- Model: 'enhanced_bitcoin_predictor.pkl'")
    
    # Print summary statistics
    print(f"\nModel Performance Summary:")
    print(f"- Dataset size: {len(predictor.feature_df)} samples")
    print(f"- Number of features: {len(predictor.feature_columns)}")
    print(f"- Prediction classes: {sorted(np.unique(predictions))}")
    print(f"- Average prediction confidence: {np.mean(max_probs):.3f}")
    
except FileNotFoundError as e:
    print(f"Error: Could not find data file. Please check your file paths.")
    print(f"Expected files: {list(data_paths.values())}")
    print(f"Error details: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

