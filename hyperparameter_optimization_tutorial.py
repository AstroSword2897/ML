# HYPERPARAMETER OPTIMIZATION TUTORIAL - STEP BY STEP
# This tutorial will teach you the fundamentals of hyperparameter optimization

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time

print("🎯 HYPERPARAMETER OPTIMIZATION TUTORIAL")
print("=" * 50)

# ============================================================================
# STEP 1: Understanding What Hyperparameters Are
# ============================================================================
print("\n📚 STEP 1: What are Hyperparameters?")
print("-" * 30)
print("Hyperparameters are settings you choose BEFORE training your model.")
print("Examples:")
print("  • k in KNN (how many neighbors to look at)")
print("  • Learning rate in neural networks")
print("  • Number of trees in Random Forest")
print("  • C and gamma in SVM")
print("These are NOT learned from data - YOU choose them!")

# ============================================================================
# STEP 2: Load and Prepare Data
# ============================================================================
print("\n📊 STEP 2: Loading Data")
print("-" * 30)
iris = load_iris()
X = iris.data  # Features (measurements)
y = iris.target  # Target (flower type)

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ============================================================================
# STEP 3: Manual Hyperparameter Tuning (The Hard Way)
# ============================================================================
print("\n🔧 STEP 3: Manual Hyperparameter Tuning")
print("-" * 30)
print("This is what you were doing before - testing each value manually")
print("Pros: Simple to understand")
print("Cons: Time-consuming, doesn't scale well")

k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
manual_results = []

print("\nTesting different k values manually:")
for k in k_values:
    # Create model with this k value
    model = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Test the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    manual_results.append(accuracy)
    
    print(f"  k={k:2d} → Accuracy: {accuracy:.3f}")

# Find best k manually
best_k_manual = k_values[np.argmax(manual_results)]
best_acc_manual = max(manual_results)
print(f"\n✅ Best k (manual): {best_k_manual} with accuracy: {best_acc_manual:.3f}")

# ============================================================================
# STEP 4: GridSearchCV (The Smart Way)
# ============================================================================
print("\n🚀 STEP 4: GridSearchCV - Automated Grid Search")
print("-" * 30)
print("GridSearchCV automatically tests ALL combinations of parameters")
print("It also uses cross-validation for more reliable results")

# Define the parameter grid
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'weights': ['uniform', 'distance'],  # How to weight neighbors
    'metric': ['euclidean', 'manhattan']  # How to measure distance
}

print(f"Parameter grid: {param_grid}")
print("This will test: 10 k values × 2 weights × 2 metrics = 40 combinations!")

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),  # The model to optimize
    param_grid=param_grid,            # Parameters to try
    cv=5,                             # 5-fold cross-validation
    scoring='accuracy',               # What to optimize for
    n_jobs=-1,                        # Use all CPU cores
    verbose=1                         # Show progress
)

# Run the search
print("\n🔍 Running GridSearchCV...")
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"\n✅ GridSearchCV completed in {grid_time:.2f} seconds")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
print(f"Test accuracy: {grid_search.score(X_test, y_test):.3f}")

# ============================================================================
# STEP 5: RandomizedSearchCV (The Fast Way)
# ============================================================================
print("\n⚡ STEP 5: RandomizedSearchCV - Random Search")
print("-" * 30)
print("RandomizedSearchCV randomly samples parameter combinations")
print("Often finds good solutions faster than grid search")

from scipy.stats import randint

# Define parameter distributions (can include continuous values)
param_dist = {
    'n_neighbors': randint(1, 20),  # Random integer between 1-19
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

print(f"Parameter distributions: {param_dist}")
print("This will randomly sample 20 combinations (much faster!)")

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=param_dist,
    n_iter=20,                       # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,                 # For reproducible results
    verbose=1
)

# Run the search
print("\n🎲 Running RandomizedSearchCV...")
start_time = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print(f"\n✅ RandomizedSearchCV completed in {random_time:.2f} seconds")
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.3f}")
print(f"Test accuracy: {random_search.score(X_test, y_test):.3f}")

# ============================================================================
# STEP 6: Compare All Methods
# ============================================================================
print("\n📊 STEP 6: Comparing All Methods")
print("-" * 30)

comparison = {
    'Manual Search': best_acc_manual,
    'GridSearchCV': grid_search.score(X_test, y_test),
    'RandomizedSearchCV': random_search.score(X_test, y_test)
}

print("Final Results:")
for method, accuracy in comparison.items():
    print(f"  {method}: {accuracy:.3f}")

best_method = max(comparison, key=comparison.get)
print(f"\n🏆 Best method: {best_method}")

# ============================================================================
# STEP 7: Visualization
# ============================================================================
print("\n📈 STEP 7: Visualization")
print("-" * 30)

plt.figure(figsize=(12, 4))

# Plot 1: Manual search results
plt.subplot(1, 3, 1)
plt.plot(k_values, manual_results, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_k_manual, color='red', linestyle='--', linewidth=2, label=f'Best k={best_k_manual}')
plt.title('Manual Search Results')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Method comparison
plt.subplot(1, 3, 2)
methods = list(comparison.keys())
accuracies = list(comparison.values())
colors = ['lightblue', 'lightgreen', 'lightcoral']
bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
plt.title('Method Comparison')
plt.ylabel('Test Accuracy')
plt.xticks(rotation=45)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.3f}', ha='center', va='bottom')

# Plot 3: Time comparison
plt.subplot(1, 3, 3)
times = [0, grid_time, random_time]  # Manual time is negligible
time_methods = ['Manual', 'GridSearch', 'RandomSearch']
plt.bar(time_methods, times, color=['orange', 'purple', 'green'], alpha=0.7)
plt.title('Time Comparison')
plt.ylabel('Time (seconds)')
for i, (method, time_val) in enumerate(zip(time_methods, times)):
    plt.text(i, time_val + 0.1, f'{time_val:.2f}s', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 8: Key Takeaways
# ============================================================================
print("\n🎓 STEP 8: Key Takeaways")
print("-" * 30)
print("1. Manual tuning: Good for learning, bad for real projects")
print("2. GridSearchCV: Tests all combinations, guaranteed best result")
print("3. RandomizedSearchCV: Faster, often finds good solutions")
print("4. Cross-validation: More reliable than single train/test split")
print("5. Always compare multiple methods!")

print("\n💡 When to use each method:")
print("  • Manual: Learning/understanding")
print("  • GridSearch: Small parameter spaces, need best result")
print("  • RandomizedSearch: Large parameter spaces, need speed")

print("\n🎉 Congratulations! You now understand hyperparameter optimization!")
print("=" * 50)
