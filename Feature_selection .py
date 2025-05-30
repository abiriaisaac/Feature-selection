# -*- coding: utf-8 -*-
"""
Script header with creation date and author information
Created on Fri Nov 15 09:43:27 2024
@author: Abiria_Isaac
"""

# Import necessary libraries
import shap  # For SHAP value calculations
import xgboost as xgb  # XGBoost machine learning model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Evaluation metrics
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Enhanced visualization
import numpy as np  # Numerical operations

# Set global plotting parameters
plt.rcParams.update({
    'font.size': 14,       # Base font size
    'font.weight': 'bold', # Bold text
    'font.family': 'serif' # Serif font family
})

# Configure seaborn style for plots
sns.set_theme(style="whitegrid")  # White background with grid
plt.rcParams.update({
    "font.size": 10,        # Default font size
    "axes.titlesize": 12,   # Title size
    "axes.labelsize": 10,   # Axis label size
    "xtick.color": "black", # X-axis tick color
    "ytick.color": "black", # Y-axis tick color
    "axes.edgecolor": "black",  # Axis edge color
    "axes.labelcolor": "black", # Label color
    "text.usetex": False     # Disable LaTeX text rendering
})

# Load dataset from CSV file
data = pd.read_csv('Feature_selection_1_2.csv')

# Display preview of the dataset
print("Dataset preview:")
print(data.head())

# Select features for analysis
X = data[['stress', 'defect_size', 'distance_from_surface', 'defect_circularity']]

# Calculate Pearson correlation matrix for features
correlation_matrix = X.corr()

# Create heatmap visualization of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={'shrink': .8})
plt.show()

# Prepare features and target variable
X = data[['stress', 'defect_size', 'distance_from_surface', 'defect_circularity']]
y = data['cycle']

# Display correlation matrix again (duplicate code in original)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={'shrink': .8})
plt.show()

# Initialize and train XGBoost regression model
model = xgb.XGBRegressor(
    learning_rate=0.1,      # Learning rate
    n_estimators=200,       # Number of trees
    max_depth=6,           # Maximum tree depth
    min_child_weight=1,    # Minimum sum of instance weight needed in a child
    subsample=0.8,         # Subsample ratio of training instances
    colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
    reg_alpha=0.1,         # L1 regularization term
    reg_lambda=1,          # L2 regularization term
    objective='reg:squarederror'  # Regression objective
)
model.fit(X, y)  # Train model

# Make predictions on training data
y_pred = model.predict(X)

# Calculate evaluation metrics
mse = mean_squared_error(y, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)                  # Root Mean Squared Error
mae = mean_absolute_error(y, y_pred) # Mean Absolute Error
r2 = r2_score(y, y_pred)            # R-squared coefficient

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R²): {r2:.4f}')

# Initialize SHAP explainer for model interpretation
explainer = shap.Explainer(model, X)

# Calculate SHAP values for all instances
shap_values = explainer(X)

# Reapply styling (duplicate in original)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.usetex": False
})

# Create SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False)
plt.gcf().get_axes()[-1].set_ylabel('Feature Value')
plt.tight_layout()
plt.show()

# Create SHAP bar plot for feature importance
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=10)
plt.show()

# Create combined SHAP plots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Left plot: SHAP summary
plt.sca(axes[0])
shap.summary_plot(shap_values, X, show=False)
axes[0].get_figure().get_axes()[-1].set_ylabel('Feature Value')

# Right plot: SHAP feature importance
plt.sca(axes[1])
shap.plots.bar(shap_values, max_display=10, show=False)

plt.subplots_adjust(wspace=0.4)
plt.tight_layout()
plt.show()

# Convert SHAP values to DataFrame
shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)

# Calculate absolute SHAP value contributions
total_contributions = shap_values_df.abs().sum(axis=0)

# Calculate percentage contributions
percentage_contributions = (total_contributions / total_contributions.sum()) * 100

# Display feature contributions
print("\nFeature Contributions in Percentage:")
print(percentage_contributions)

# Save contributions to Excel
output_df = pd.DataFrame({
    'Feature': percentage_contributions.index,
    'Percentage Contribution': percentage_contributions.values
})
output_file = 'feature_contributions.xlsx'
output_df.to_excel(output_file, index=False)
print(f"Feature contributions saved to {output_file}")

# Create pie chart of feature contributions
plt.figure(figsize=(8, 8))
percentage_contributions.plot(kind='pie', autopct='%1.1f%%', startangle=90, colormap='tab10')
plt.ylabel("")
plt.title("Feature Contributions to Predictions (%)")
plt.tight_layout()
plt.show()

# Duplicate code block from original (same as above)
shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
total_contributions = shap_values_df.abs().sum(axis=0)
percentage_contributions = (total_contributions / total_contributions.sum()) * 100

print("\nFeature Contributions in Percentage:")
print(percentage_contributions)

output_df = pd.DataFrame({
    'Feature': percentage_contributions.index,
    'Percentage Contribution': percentage_contributions.values
})
output_file = 'feature_contributions.xlsx'
output_df.to_excel(output_file, index=False)
print(f"Feature contributions saved to {output_file}")

# Enhanced pie chart visualization
plt.figure(figsize=(10, 10))
colors = sns.color_palette("tab10", len(percentage_contributions))
percentage_contributions.plot(
    kind='pie', 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=colors,
    wedgeprops={"edgecolor": "black", "linewidth": 1}
)
plt.title("Feature Contributions to Fatigue life Predictions (%)", fontsize=12, fontweight="bold")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Save pie chart as image
plt.savefig('feature_contributions_piechart.png', dpi=300, bbox_inches='tight')
