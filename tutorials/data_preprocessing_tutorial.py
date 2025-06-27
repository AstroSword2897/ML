# 📊 DATA PREPROCESSING TUTORIAL
# Essential techniques for cleaning and preparing data for ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("🧹 DATA PREPROCESSING TUTORIAL")
print("=" * 50)

# ============================================================================
# STEP 1: Understanding Data Preprocessing
# ============================================================================
print("\n📚 STEP 1: What is Data Preprocessing?")
print("-" * 30)
print("Data preprocessing is cleaning and transforming raw data")
print("into a format that ML algorithms can understand.")
print("\nKey steps:")
print("  • Handling missing values")
print("  • Dealing with outliers")
print("  • Encoding categorical variables")
print("  • Feature scaling")
print("  • Feature engineering")

# ============================================================================
# STEP 2: Create Sample Dataset with Common Issues
# ============================================================================
print("\n📊 STEP 2: Creating Sample Dataset with Common Data Issues")
print("-" * 30)

# Create a realistic dataset with common problems
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.normal(35, 10, n_samples),
    'salary': np.random.normal(50000, 15000, n_samples),
    'experience_years': np.random.normal(8, 4, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], n_samples),
    'performance_score': np.random.normal(75, 15, n_samples),
    'satisfaction': np.random.choice([1, 2, 3, 4, 5], n_samples),
    'remote_work': np.random.choice([True, False], n_samples)
}

df = pd.DataFrame(data)

# Introduce common data issues
print("Introducing common data problems...")

# 1. Missing values
df.loc[np.random.choice(df.index, 50), 'age'] = np.nan
df.loc[np.random.choice(df.index, 30), 'salary'] = np.nan
df.loc[np.random.choice(df.index, 20), 'education'] = np.nan

# 2. Outliers
df.loc[0, 'salary'] = 500000  # Extreme outlier
df.loc[1, 'age'] = 150        # Impossible age

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# STEP 3: Data Exploration
# ============================================================================
print("\n🔍 STEP 3: Data Exploration")
print("-" * 30)

print("1. Basic Information:")
print(df.info())

print("\n2. Missing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\n3. Statistical Summary:")
print(df.describe())

# ============================================================================
# STEP 4: Handling Missing Values
# ============================================================================
print("\n🔧 STEP 4: Handling Missing Values")
print("-" * 30)

print("Before cleaning:")
print(f"Missing values: {df.isnull().sum().sum()}")

# Create a copy for preprocessing
df_clean = df.copy()

# Fill missing values
df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
df_clean['salary'].fillna(df_clean['salary'].median(), inplace=True)
df_clean['education'].fillna(df_clean['education'].mode()[0], inplace=True)

print("After filling missing values:")
print(f"Missing values: {df_clean.isnull().sum().sum()}")

# ============================================================================
# STEP 5: Handling Outliers
# ============================================================================
print("\n📈 STEP 5: Handling Outliers")
print("-" * 30)

# Handle outliers by capping them
for col in ['age', 'salary', 'experience_years', 'performance_score']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

print("Outliers handled by capping values")

# ============================================================================
# STEP 6: Encoding Categorical Variables
# ============================================================================
print("\n🔤 STEP 6: Encoding Categorical Variables")
print("-" * 30)

# One-Hot Encoding for categorical variables
education_dummies = pd.get_dummies(df_clean['education'], prefix='education')
department_dummies = pd.get_dummies(df_clean['department'], prefix='dept')

# Add encoded columns to dataframe
df_clean = pd.concat([df_clean, education_dummies, department_dummies], axis=1)

# Remove original categorical columns
df_clean.drop(['education', 'department', 'satisfaction'], axis=1, inplace=True)

print("Categorical variables encoded")

# ============================================================================
# STEP 7: Feature Scaling
# ============================================================================
print("\n⚖️ STEP 7: Feature Scaling")
print("-" * 30)

# Identify numerical columns for scaling
numerical_cols = ['age', 'salary', 'experience_years', 'performance_score']

# Apply StandardScaler
scaler = StandardScaler()
df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

print("Features scaled using StandardScaler")

# ============================================================================
# STEP 8: Final Dataset Preparation
# ============================================================================
print("\n🎯 STEP 8: Final Dataset Preparation")
print("-" * 30)

# Create target variable (example: high performer)
df_clean['high_performer'] = (df_clean['performance_score'] > df_clean['performance_score'].median()).astype(int)

# Select features for modeling
feature_cols = [col for col in df_clean.columns if col not in ['high_performer']]

X = df_clean[feature_cols]
y = df_clean['high_performer']

print(f"Final dataset shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================================
# STEP 9: Quick Model Test
# ============================================================================
print("\n🤖 STEP 9: Quick Model Test")
print("-" * 30)

# Train a simple model to test our preprocessing
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# ============================================================================
# STEP 10: Key Takeaways
# ============================================================================
print("\n🎓 STEP 10: Key Takeaways")
print("-" * 30)
print("✅ Data preprocessing is crucial for ML success")
print("✅ Always explore your data before preprocessing")
print("✅ Handle missing values appropriately")
print("✅ Detect and handle outliers")
print("✅ Encode categorical variables properly")
print("✅ Scale numerical features")
print("✅ Test your preprocessing with a model")

print("\n💡 Best Practices:")
print("  • Keep original data as backup")
print("  • Document all preprocessing steps")
print("  • Use cross-validation for reliable results")
print("  • Monitor data quality continuously")

print("\n🎉 Congratulations! You've mastered data preprocessing fundamentals!")
print("=" * 50) 