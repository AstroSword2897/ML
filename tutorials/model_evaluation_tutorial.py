# 📊 MODEL EVALUATION TUTORIAL
# Comprehensive guide to evaluating machine learning models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

print("📊 MODEL EVALUATION TUTORIAL")
print("=" * 50)

# ============================================================================
# STEP 1: Understanding Model Evaluation
# ============================================================================
print("\n📚 STEP 1: What is Model Evaluation?")
print("-" * 30)
print("Model evaluation is the process of assessing how well a model")
print("performs on unseen data and understanding its strengths/weaknesses.")
print("\nKey aspects:")
print("  • Performance metrics")
print("  • Cross-validation")
print("  • Bias vs Variance")
print("  • Overfitting detection")
print("  • Model comparison")

# ============================================================================
# STEP 2: Load and Prepare Data
# ============================================================================
print("\n📊 STEP 2: Loading and Preparing Data")
print("-" * 30)

# Load two different datasets for comparison
iris = load_iris()
breast_cancer = load_breast_cancer()

# Use binary classification for breast cancer
X_bc = breast_cancer.data
y_bc = breast_cancer.target

# Use multi-class for iris (first two classes for simplicity)
X_iris = iris.data[iris.target != 2]
y_iris = iris.target[iris.target != 2]

print(f"Breast Cancer Dataset: {X_bc.shape}")
print(f"Iris Dataset (Binary): {X_iris.shape}")

# Split data
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# ============================================================================
# STEP 3: Train Multiple Models
# ============================================================================
print("\n🤖 STEP 3: Training Multiple Models")
print("-" * 30)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Train models on breast cancer dataset
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_bc, y_train_bc)
    trained_models[name] = model

# ============================================================================
# STEP 4: Basic Classification Metrics
# ============================================================================
print("\n📈 STEP 4: Basic Classification Metrics")
print("-" * 30)

print("Understanding Classification Metrics:")
print("• Accuracy: Overall correctness")
print("• Precision: How many predicted positives were actually positive")
print("• Recall: How many actual positives were correctly predicted")
print("• F1-Score: Harmonic mean of precision and recall")

results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test_bc)
    y_pred_proba = model.predict_proba(X_test_bc)[:, 1]
    
    results[name] = {
        'accuracy': accuracy_score(y_test_bc, y_pred),
        'precision': precision_score(y_test_bc, y_pred),
        'recall': recall_score(y_test_bc, y_pred),
        'f1': f1_score(y_test_bc, y_pred),
        'auc': roc_auc_score(y_test_bc, y_pred_proba),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# Display results
print("\nModel Performance Comparison:")
print("-" * 50)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    print(f"  AUC:       {metrics['auc']:.3f}")

# ============================================================================
# STEP 5: Confusion Matrix
# ============================================================================
print("\n🔍 STEP 5: Confusion Matrix Analysis")
print("-" * 30)

print("Confusion Matrix shows:")
print("• True Negatives (TN): Correctly predicted negative")
print("• False Positives (FP): Incorrectly predicted positive")
print("• False Negatives (FN): Incorrectly predicted negative")
print("• True Positives (TP): Correctly predicted positive")

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices for Different Models')

for i, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_bc)
    cm = confusion_matrix(y_test_bc, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 6: ROC Curve and AUC
# ============================================================================
print("\n📊 STEP 6: ROC Curve and AUC")
print("-" * 30)

print("ROC Curve shows:")
print("• True Positive Rate (Sensitivity) vs False Positive Rate (1-Specificity)")
print("• AUC = Area Under Curve (higher is better)")
print("• Perfect classifier has AUC = 1.0")
print("• Random classifier has AUC = 0.5")

plt.figure(figsize=(10, 6))

for name, metrics in results.items():
    fpr, tpr, _ = roc_curve(y_test_bc, metrics['probabilities'])
    auc = metrics['auc']
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# STEP 7: Precision-Recall Curve
# ============================================================================
print("\n📈 STEP 7: Precision-Recall Curve")
print("-" * 30)

print("Precision-Recall Curve is useful when:")
print("• Classes are imbalanced")
print("• False positives are costly")
print("• You want to focus on positive class")

plt.figure(figsize=(10, 6))

for name, metrics in results.items():
    precision, recall, _ = precision_recall_curve(y_test_bc, metrics['probabilities'])
    plt.plot(recall, precision, label=f'{name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# STEP 8: Cross-Validation
# ============================================================================
print("\n🔄 STEP 8: Cross-Validation")
print("-" * 30)

print("Cross-Validation benefits:")
print("• More reliable performance estimate")
print("• Reduces overfitting risk")
print("• Better use of limited data")
print("• More robust model selection")

# Perform cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model in models.items():
    print(f"\nPerforming 5-fold CV for {name}...")
    cv_scores = cross_val_score(model, X_bc, y_bc, cv=cv, scoring='accuracy')
    cv_results[name] = cv_scores
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Plot CV results
plt.figure(figsize=(10, 6))
cv_df = pd.DataFrame(cv_results)
cv_df.boxplot()
plt.title('Cross-Validation Results')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# STEP 9: Model Comparison and Selection
# ============================================================================
print("\n🏆 STEP 9: Model Comparison and Selection")
print("-" * 30)

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'Precision': [results[name]['precision'] for name in results.keys()],
    'Recall': [results[name]['recall'] for name in results.keys()],
    'F1-Score': [results[name]['f1'] for name in results.keys()],
    'AUC': [results[name]['auc'] for name in results.keys()],
    'CV_Mean': [cv_results[name].mean() for name in results.keys()],
    'CV_Std': [cv_results[name].std() for name in results.keys()]
})

print("Comprehensive Model Comparison:")
print(comparison_df.round(3))

# Find best model based on different criteria
best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
best_auc = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']
best_cv = comparison_df.loc[comparison_df['CV_Mean'].idxmax(), 'Model']

print(f"\nBest Models by Different Criteria:")
print(f"  • Highest Accuracy: {best_accuracy}")
print(f"  • Highest F1-Score: {best_f1}")
print(f"  • Highest AUC: {best_auc}")
print(f"  • Best CV Performance: {best_cv}")

# ============================================================================
# STEP 10: Detailed Classification Report
# ============================================================================
print("\n📋 STEP 10: Detailed Classification Report")
print("-" * 30)

# Show detailed report for best model
best_model_name = best_f1  # Using F1-score as it balances precision and recall
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test_bc)

print(f"Detailed Classification Report for {best_model_name}:")
print(classification_report(y_test_bc, y_pred_best, target_names=['Benign', 'Malignant']))

# ============================================================================
# STEP 11: Visualization Summary
# ============================================================================
print("\n📊 STEP 11: Visualization Summary")
print("-" * 30)

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model comparison bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
x = np.arange(len(metrics))
width = 0.25

for i, (name, color) in enumerate(zip(results.keys(), ['blue', 'orange', 'green'])):
    values = [results[name][metric.lower().replace('-', '')] for metric in metrics]
    axes[0, 0].bar(x + i*width, values, width, label=name, color=color, alpha=0.7)

axes[0, 0].set_xlabel('Metrics')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Model Performance Comparison')
axes[0, 0].set_xticks(x + width)
axes[0, 0].set_xticklabels(metrics)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. CV results
cv_df.boxplot(ax=axes[0, 1])
axes[0, 1].set_title('Cross-Validation Results')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. ROC curves
for name, metrics in results.items():
    fpr, tpr, _ = roc_curve(y_test_bc, metrics['probabilities'])
    auc = metrics['auc']
    axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curves')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Precision-Recall curves
for name, metrics in results.items():
    precision, recall, _ = precision_recall_curve(y_test_bc, metrics['probabilities'])
    axes[1, 1].plot(recall, precision, label=name)

axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title('Precision-Recall Curves')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 12: Key Takeaways
# ============================================================================
print("\n🎓 STEP 12: Key Takeaways")
print("-" * 30)
print("✅ Always use multiple metrics for evaluation")
print("✅ Cross-validation provides more reliable estimates")
print("✅ ROC curves are great for balanced datasets")
print("✅ Precision-Recall curves are better for imbalanced data")
print("✅ Confusion matrices help understand model behavior")
print("✅ Choose metrics based on business requirements")

print("\n💡 When to use which metric:")
print("  • Accuracy: Balanced classes, equal costs")
print("  • Precision: False positives are costly")
print("  • Recall: False negatives are costly")
print("  • F1-Score: Balance between precision and recall")
print("  • AUC: Overall model performance")

print("\n🚀 Next Steps:")
print("  • Learn about hyperparameter tuning")
print("  • Explore ensemble methods")
print("  • Study model interpretability")
print("  • Practice with real-world datasets")

print("\n🎉 Congratulations! You've mastered model evaluation!")
print("=" * 50) 