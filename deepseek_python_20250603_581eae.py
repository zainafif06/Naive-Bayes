"""
Breast Cancer Prediction using Wisconsin Diagnostic Dataset
Author: [Your Name]
Date: [Current Date]
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV

# Create directory for saving images
os.makedirs('images', exist_ok=True)

def save_plot(fig, filename):
    """Helper function to save plots"""
    path = os.path.join('images', filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path}")

def main():
    # Load dataset
    print("Loading dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(np.c_[data['data'], data['target']], 
                      columns=np.append(data['feature_names'], ['target']))
    
    # 1. Data Exploration
    print("\n=== Data Exploration ===")
    print(f"Dataset shape: {df.shape}")
    print("\nTarget distribution:")
    print(df['target'].value_counts())
    
    # Plot class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='target', data=df, ax=ax)
    ax.set_title('Class Distribution (0: Malignant, 1: Benign)')
    save_plot(fig, 'class_distribution.png')
    
    # 2. Data Preprocessing
    print("\nPreprocessing data...")
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Training and Evaluation
    print("\nTraining models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        results.append(metrics)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        save_plot(fig, f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        
        # Classification report
        print(f"\nClassification Report - {name}:")
        print(classification_report(y_test, y_pred))
    
    # 4. Results Comparison
    results_df = pd.DataFrame(results).set_index('Model')
    print("\n=== Model Performance Comparison ===")
    print(results_df.round(3))
    
    # Save results to CSV
    results_df.to_csv('model_results.csv')
    print("\nSaved model results to 'model_results.csv'")
    
    # 5. ROC Curve Comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    save_plot(fig, 'roc_curve_comparison.png')
    
    # 6. Feature Importance (using Random Forest)
    print("\nAnalyzing feature importance...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    feature_importance = rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Plot top 10 features
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', 
                data=importance_df.head(10), ax=ax)
    ax.set_title('Top 10 Important Features')
    save_plot(fig, 'feature_importance.png')
    
    # 7. Hyperparameter Tuning (Optional)
    print("\nPerforming hyperparameter tuning for Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test_scaled)
    print("\nOptimized Random Forest Performance:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
    print("\nAnalysis complete! Check the 'images' folder for visualizations.")