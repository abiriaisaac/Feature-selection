
"""
XGBoost and SHAP Analysis

This script performs regression using XGBoost and analyzes feature importance using SHAP.
"""

# Import necessary libraries
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create outputs directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Set up plot styles
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'font.family': 'serif'
})
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

# Load dataset
data = pd.read_csv('Feature_selection_1_2.csv')#Load based on the data source to be analysed
print("Dataset preview:")
print(data.head())

# Separate features and target variable
X = data[['stress', 'defect_size', 'distance_from_surface', 'defect_circularity']]#The input features of the data to be analysed
y = data['cycle']#Targeted output

# Plot correlation matrix for feature dependence analysis
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={'shrink': .8})
plt.title("Pearson Correlation Coefficient Heatmap")
plt.show()

# Train XGBoost model #Note XGboost is the model used
model = xgb.XGBRegressor(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    objective='reg:squarederror'
)
model.fit(X, y)

# Evaluate model
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R²): {r2:.4f}')

# SHAP analysis
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False)
plt.gcf().get_axes()[-1].set_ylabel('Feature Value')
plt.tight_layout()
plt.show()

# SHAP bar plot
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=10)
plt.show()

# Combined SHAP summary and bar plots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
plt.sca(axes[0])
shap.summary_plot(shap_values, X, show=False)
axes[0].get_figure().get_axes()[-1].set_ylabel('Feature Value')
plt.sca(axes[1])
shap.plots.bar(shap_values, max_display=10, show=False)
plt.subplots_adjust(wspace=0.4)
plt.tight_layout()
plt.show()

# Calculate feature contributions
shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
total_contributions = shap_values_df.abs().sum(axis=0)
percentage_contributions = (total_contributions / total_contributions.sum()) * 100

# Save feature contributions to Excel
output_df = pd.DataFrame({
    'Feature': percentage_contributions.index,
    'Percentage Contribution': percentage_contributions.values
})
output_file = 'outputs/feature_contributions.xlsx'
output_df.to_excel(output_file, index=False)
print(f"Feature contributions saved to {output_file}")

# Plot feature contributions as a pie chart
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
plt.savefig('outputs/feature_contributions_piechart.png', dpi=300, bbox_inches='tight')