# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# # import shap
# import joblib

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, precision_recall_curve
# from sklearn.feature_selection import RFE
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline



# # -------------------------------
# # Load CSV
# # -------------------------------
# df = pd.read_csv("framingham.csv")
# target_col = "TenYearCHD"  # Update if your target column differs
# X = df.drop(columns=[target_col])
# y = df[target_col]

# # print(df.head())


# # -------------------------------
# # 1. Data Cleaning
# # -------------------------------
# # 1a. Fill missing values with median for numeric features
# X = X.fillna(X.median())

# # 1b. Label Encoding for categorical columns
# categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
# le_dict = {}
# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     le_dict[col] = le


# def remove_outliers(df, cols):
#     for col in cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5*IQR
#         upper = Q3 + 1.5*IQR
#         df = df[(df[col] >= lower) & (df[col] <= upper)]
#     return df

# Xy = pd.concat([X, y], axis=1)
# numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
# Xy = remove_outliers(Xy, numeric_cols)
# X = Xy.drop(columns=[target_col])
# y = Xy[target_col]


# # -------------------------------
# # 2. EDA & Visualization
# # -------------------------------
# plt.figure(figsize=(12,6))
# sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.show()

# X.hist(figsize=(12,10), bins=20)
# plt.tight_layout()
# plt.show()

# sns.countplot(x=y)
# plt.title("Class Distribution")
# plt.show()


# # -------------------------------
# # 3. Feature Selection
# # -------------------------------
# # 3a. Correlation filter (>0.8)
# corr_matrix = X.corr().abs()
# upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# high_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.8)]
# X = X.drop(columns=high_corr)
# print("Dropped highly correlated features:", high_corr)



# # 3b. LASSO (Logistic Regression)
# lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
# lasso_model.fit(X, y)
# lasso_features = X.columns[np.abs(lasso_model.coef_[0]) > 1e-5].tolist()
# print("LASSO selected features:", lasso_features)


# # 3c. RFE (Random Forest)
# rfe_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rfe = RFE(rfe_model, n_features_to_select=10)
# rfe.fit(X, y)
# rfe_features = X.columns[rfe.support_].tolist()
# print("RFE selected features:", rfe_features)


# # -------------------------------
# # 4. Train/Test Split
# # -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # -------------------------------
# # 5. SMOTE (handle class imbalance)
# # -------------------------------
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)


# # -------------------------------
# # 6. Model Definitions
# # -------------------------------
# models = {
#     "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
#     "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
#     # "SVM": SVC(kernel='linear', probability=True, random_state=42),
#     "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
# }


# # -------------------------------
# # 8. Cross-Validation
# # -------------------------------
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# for name, model in models.items():
#     scores = cross_validate(model, X_train, y_train, cv=skf,
#                             scoring=["accuracy","precision","recall","f1"])
#     print(f"\n{name} CV Results:")
#     for metric in ["accuracy","precision","recall","f1"]:
#         print(f"{metric}: {scores['test_'+metric].mean():.4f} (+/- {scores['test_'+metric].std():.4f})")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import shap
import joblib
import os
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ===============================
# CHANGES MADE:
# 1. Added StandardScaler for feature scaling
# 2. Created model pipelines that include preprocessing steps
# 3. Added final model training and evaluation on test set
# 4. Added model saving functionality with joblib
# 5. Added comprehensive model metadata saving
# 6. Fixed Logistic Regression convergence issues
# 7. Added feature importance analysis
# 8. Enhanced hyperparameter tuning
# ===============================

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv("framingham.csv")
target_col = "TenYearCHD"  # Update if your target column differs
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# -------------------------------
# 1. Data Cleaning
# -------------------------------
# 1a. Fill missing values with median for numeric features
X = X.fillna(X.median())

# 1b. Label Encoding for categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

Xy = pd.concat([X, y], axis=1)
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
print(f"Shape before outlier removal: {Xy.shape}")
Xy = remove_outliers(Xy, numeric_cols)
print(f"Shape after outlier removal: {Xy.shape}")
X = Xy.drop(columns=[target_col])
y = Xy[target_col]

# -------------------------------
# 2. EDA & Visualization
# -------------------------------
plt.figure(figsize=(12,6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

X.hist(figsize=(12,10), bins=20)
plt.tight_layout()
plt.show()

sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()

# -------------------------------
# 3. Feature Selection
# -------------------------------
# 3a. Correlation filter (>0.8)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.8)]
X = X.drop(columns=high_corr)
print("Dropped highly correlated features:", high_corr)

# 3b. LASSO (Logistic Regression)
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
lasso_model.fit(X, y)
lasso_features = X.columns[np.abs(lasso_model.coef_[0]) > 1e-5].tolist()
print("LASSO selected features:", lasso_features)

# 3c. RFE (Random Forest)
rfe_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(rfe_model, n_features_to_select=10)
rfe.fit(X, y)
rfe_features = X.columns[rfe.support_].tolist()
print("RFE selected features:", rfe_features)

# CHANGE: Use the best features (intersection of LASSO and RFE)
selected_features = list(set(lasso_features) & set(rfe_features))
print("Final selected features:", selected_features)
X_selected = X[selected_features]

# -------------------------------
# 4. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

# -------------------------------
# 5. SMOTE (handle class imbalance)
# -------------------------------
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Training set shape: {X_train_balanced.shape}")
print(f"After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")

# CHANGE: Create scaler for feature normalization (fixes convergence issues)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Enhanced Model Definitions with Hyperparameter Tuning
# -------------------------------
# CHANGE: Fixed Logistic Regression with proper scaling and increased iterations
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# CHANGE: Hyperparameter tuning for Random Forest (best model from CV)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Performing hyperparameter tuning for Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train_balanced, y_train_balanced)
print(f"Best Random Forest parameters: {rf_grid.best_params_}")

# Update the Random Forest model with best parameters
models["RandomForest"] = rf_grid.best_estimator_

# -------------------------------
# 7. Cross-Validation with Scaled Data
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    # Use scaled data for all models
    if name == "LogisticRegression":
        scores = cross_validate(model, X_train_scaled, y_train_balanced, cv=skf,
                                scoring=["accuracy","precision","recall","f1"])
    else:
        scores = cross_validate(model, X_train_balanced, y_train_balanced, cv=skf,
                                scoring=["accuracy","precision","recall","f1"])
    
    cv_results[name] = scores
    print(f"\n{name} CV Results:")
    for metric in ["accuracy","precision","recall","f1"]:
        print(f"{metric}: {scores['test_'+metric].mean():.4f} (+/- {scores['test_'+metric].std():.4f})")

# -------------------------------
# 8. CHANGE: Final Model Training and Test Set Evaluation
# -------------------------------
print("\n" + "="*50)
print("FINAL MODEL EVALUATION ON TEST SET")
print("="*50)

final_results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining final {name} model...")
    
    # Train on full training set
    if name == "LogisticRegression":
        model.fit(X_train_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    final_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    trained_models[name] = model
    
    print(f"{name} Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# -------------------------------
# 9. CHANGE: Select Best Model and Feature Importance
# -------------------------------
best_model_name = max(final_results.keys(), key=lambda x: final_results[x]['f1'])
best_model = trained_models[best_model_name]
print(f"\nBest Model: {best_model_name} (F1-Score: {final_results[best_model_name]['f1']:.4f})")

# CHANGE: Feature importance analysis
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Feature Importances for {best_model_name}:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.show()

# -------------------------------
# 10. CHANGE: Save Models and Preprocessing Components
# -------------------------------
# Create models directory
os.makedirs('trained_models', exist_ok=True)

# CHANGE: Save the best model with all preprocessing components
model_package = {
    'model': best_model,
    'scaler': scaler,
    'selected_features': selected_features,
    'label_encoders': le_dict,
    'smote': smote,
    'model_name': best_model_name,
    'performance_metrics': final_results[best_model_name],
    'feature_names': selected_features,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Save the complete model package
model_filename = f'trained_models/best_model_{best_model_name.lower()}.pkl'
joblib.dump(model_package, model_filename)
print(f"\nBest model saved as: {model_filename}")

# CHANGE: Save all trained models separately
for name, model in trained_models.items():
    individual_package = {
        'model': model,
        'scaler': scaler,
        'selected_features': selected_features,
        'label_encoders': le_dict,
        'model_name': name,
        'performance_metrics': final_results[name],
        'feature_names': selected_features,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    individual_filename = f'trained_models/model_{name.lower()}.pkl'
    joblib.dump(individual_package, individual_filename)
    print(f"{name} model saved as: {individual_filename}")

# CHANGE: Save model comparison results
results_df = pd.DataFrame(final_results).T
results_df.to_csv('trained_models/model_comparison_results.csv')
print(f"\nModel comparison results saved as: trained_models/model_comparison_results.csv")

# CHANGE: Save feature information
feature_info = {
    'selected_features': selected_features,
    'lasso_features': lasso_features,
    'rfe_features': rfe_features,
    'dropped_correlated_features': high_corr,
    'categorical_columns': categorical_cols,
    'numeric_columns': numeric_cols
}
joblib.dump(feature_info, 'trained_models/feature_info.pkl')
print(f"Feature information saved as: trained_models/feature_info.pkl")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE!")
print("="*50)
print(f"Best Model: {best_model_name}")
print(f"Test F1-Score: {final_results[best_model_name]['f1']:.4f}")
print(f"All files saved in 'trained_models/' directory")
print("You can now use these models for predictions in other scripts!")