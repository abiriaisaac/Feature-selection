# Feature-selection-Analysis
This repository uses Pearson correlation for feature independence analysis, trains an XGBoost regression model, and employs SHAP for feature importance analysis to predict material fatigue life cycles based on stress, defect size, distance from the surface, and defect circularity.

# XGBoost and SHAP Analysis

This project demonstrates the use of **XGBoost** for regression and **SHAP (SHapley Additive exPlanations)** for feature importance analysis. The goal is to predict the fatigue life cycle of a material based on features like stress, defect size, distance from the surface, and defect circularity. The project includes data loading, model training, evaluation, and interpretation using SHAP values.

---

## Features
- **Data Loading and Preprocessing**: Loads the dataset and separates features and target variables.
- **XGBoost Model Training**: Trains an XGBoost regression model.
- **Model Evaluation**: Evaluates the model using metrics like MSE, RMSE, MAE, and R².
- **SHAP Analysis**: Interprets the model using SHAP values to understand feature contributions.
- **Visualizations**: Includes correlation heatmaps, SHAP summary plots, and feature contribution pie charts.

---

## Project Structure
```
xgboost-shap-analysis/
│
├── README.md                   # Project documentation
├── requirements.txt            # List of dependencies
├── main.py                     # Main script to run the analysis
├── data/                       # Contains the dataset
│   └── Feature_selection_1_2.csv
└── outputs/                    # Stores output files
    ├── feature_contributions.xlsx
    └── feature_contributions_piechart.png
```

---

## Installation

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/abiriaisaac/xgboost-shap-analysis.git
cd xgboost-shap-analysis
```

### 2. Install Dependencies
This project requires Python 3.x and several libraries. Install the dependencies using pip or conda
#### Dependencies
The `requirements.txt` file includes the following libraries:
- `shap`: For SHAP value calculations and visualizations.
- `xgboost`: For training the XGBoost regression model.
- `scikit-learn`: For model evaluation metrics.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib`: For creating visualizations.
- `seaborn`: For enhanced visualizations.
- `openpyxl`: For saving feature contributions to an Excel file.

---

## Usage

### 1. Prepare the Dataset
Place your dataset (`Feature_selection_1_2.csv`) in the `data/` folder. The dataset should contain the following columns:
- `stress`
- `defect_size`
- `distance_from_surface`
- `defect_circularity`
- `cycle` (target variable)

### 2. Run the Script
Run the `main.py` script to perform the analysis:
```bash
python main.py
```

### 3. Check Outputs
After running the script, the following outputs will be generated:
- **`outputs/feature_contributions.xlsx`**: Excel file containing the percentage contribution of each feature.
- **`outputs/feature_contributions_piechart.png`**: Pie chart visualizing the feature contributions.

---

## Code Overview

### `main.py`
This is the main script that performs the following tasks:
1. **Loads the dataset** and separates features and target variables.
2. **Trains an XGBoost regression model**.
3. **Evaluates the model** using metrics like MSE, RMSE, MAE, and R².
4. **Performs SHAP analysis** to interpret the model and understand feature contributions.
5. **Generates visualizations** including correlation heatmaps, SHAP summary plots, and feature contribution pie charts.

---

## Outputs

### 1. **Correlation Heatmap**
A heatmap showing the Pearson correlation coefficients between features.

### 2. **Model Evaluation Metrics**
The script prints the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

### 3. **SHAP Summary Plot**
A summary plot showing the impact of each feature on the model's predictions.

### 4. **SHAP Bar Plot**
A bar plot showing the mean absolute SHAP values for the top 10 features.

### 5. **Feature Contributions**
- An Excel file (`feature_contributions.xlsx`) containing the percentage contribution of each feature.
- A pie chart (`feature_contributions_piechart.png`) visualizing the feature contributions.

---

## Example Outputs

### Correlation Heatmap
![Correlation Heatmap](outputs/correlation_heatmap.png)

### SHAP Summary Plot
![SHAP Summary Plot](outputs/shap_summary_plot.png)

### Feature Contributions Pie Chart
![Feature Contributions Pie Chart](outputs/feature_contributions_piechart.png)

---

## Troubleshooting

### 1. **Missing Dataset**
Ensure the dataset (`Feature_selection_1_2.csv`) is placed in the `data/` folder.

### 2. **Dependency Installation Issues**
If you encounter issues while installing dependencies, try upgrading `pip`:
```bash
pip install --upgrade pip
```

### 3. **File Not Found Error**
Ensure the `outputs/` folder exists. If not, create it manually:
```bash
mkdir outputs
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **XGBoost**: [https://xgboost.ai/](https://xgboost.ai/)
- **SHAP**: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
- **Scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/)

---

## How to Upload to GitHub

1. **Create a New Repository**:
   - Go to GitHub and create a new repository named `xgboost-shap-analysis`.

2. **Initialize the Repository**:
   - Open a terminal and navigate to your project folder.
   - Run the following commands to initialize the repository and push the code:
     ```bash
     git init
     git add .
     git commit -m "Initial commit"
     git branch -M main
     git remote add origin https://github.com/yourusername/xgboost-shap-analysis.git
     git push -u origin main
     ```

3. **Verify Upload**:
   - Go to your GitHub repository and ensure all files (including `README.md`, `main.py`, `requirements.txt`, and the `data/` folder) are uploaded.

---

This README file provides all the necessary information to set up, run, and understand the project. It is designed to be user-friendly and comprehensive for both beginners and experienced users.
