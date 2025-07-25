# US Health Insurance Dataset Analysis

## Overview
This project analyzes the **US Health Insurance Dataset** to predict insurance charges based on features such as age, BMI, smoking status, and more. The dataset is sourced from Kaggle and contains 1,338 entries with 7 columns. The analysis includes exploratory data analysis (EDA), statistical tests, and machine learning models to predict insurance charges. Additionally, a Streamlit app is provided to allow users to input values and receive predicted insurance charges.

## Dataset
The dataset (`insurance.csv`) is sourced from [Kaggle: US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset). It contains the following columns:

- **age**: Age of the insured (integer, 18–64)
- **sex**: Gender of the insured (object: male/female)
- **bmi**: Body Mass Index (float, 15.96–53.13)
- **children**: Number of children/dependents (integer, 0–5)
- **smoker**: Smoking status (object: yes/no)
- **region**: Region in the US (object: northeast, northwest, southeast, southwest)
- **charges**: Insurance premium charges (float, target variable)

### Dataset Summary
- **Rows**: 1,338
- **Columns**: 7
- **Missing Values**: None
- **Target Variable**: `charges` (continuous, regression problem)

## Project Structure
The Jupyter Notebook (`insurance-data-statistical-analysis (1).ipynb`) performs the following tasks:
1. **Data Loading and Exploration**:
   - Loads the dataset using pandas.
   - Checks for missing values (none found).
   - Displays data types and basic statistics.
   - Visualizes distributions of `age`, `bmi`, and `charges` using Seaborn.

2. **Problem Identification**:
   - Identifies the problem as a **regression task** to predict `charges`.
   - Confirms `charges` as the target variable (continuous numeric values).

3. **Model Training and Evaluation**:
   - Trains multiple regression models: Linear Regression, Ridge, Lasso, Decision Tree, and Random Forest.
   - Evaluates models using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
   - **Model Performance**:
     | Model                 | MAE     | MSE       | R² Score |
     |-----------------------|---------|-----------|----------|
     | Linear Regression     | 4301.33 | 35.87M    | 0.7623   |
     | Ridge                 | 4313.03 | 35.90M    | 0.7621   |
     | Lasso                 | 4301.33 | 35.87M    | 0.7623   |
     | Decision Tree         | 2917.81 | 35.45M    | 0.7650   |
     | Random Forest         | 2839.15 | 27.65M    | 0.8167   |

4. **Feature Importance**:
   - **Linear Models**: Smoking status has the highest impact (~$23,668 increase in charges), followed by age (~$3,681 per year) and BMI (~$1,991 per unit).
   - **Tree-Based Models**: Smoking status contributes ~61.3% to predictions, followed by BMI (~24.6%) and age (~14.1%).

5. **Streamlit App**:
   - A simple Streamlit app (`my_app.py`) is provided to predict insurance charges based on user inputs for `age`, `bmi`, and `smoker` status.
   - The app uses the trained Random Forest model and a saved StandardScaler for preprocessing.

## Insights
- **Smoking Status**: The most significant predictor, increasing insurance charges by ~$23,600 for smokers.
- **Age and BMI**: Age increases charges by ~$3,681 per year, and each BMI unit adds ~$1,991.
- **Region and Children**: These have minimal impact on charges compared to smoking, age, and BMI.
- **Model Choice**: Random Forest performs best (R² = 0.8167), indicating it captures non-linear relationships effectively.

## Requirements
To run the notebook and Streamlit app, install the following Python packages:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn streamlit joblib
```

## How to Run
1. **Jupyter Notebook**:
   - Ensure the dataset (`insurance.csv`) is in the correct directory (`/kaggle/input/ushealthinsurancedataset/`).
   - Run the notebook (`insurance-data-statistical-analysis.ipynb`) in a Jupyter environment to perform EDA, train models, and save the Random Forest model and scaler.

2. **Streamlit App**:
   - Save the trained model (`insurance_model.pkl`) and scaler (`scaler.pkl`) from the notebook.
   - Run the Streamlit app:
     ```bash
     streamlit run my_app.py
     ```
   - Access the app in a browser to input `age`, `bmi`, and `smoker` status and view predicted insurance charges.

## Files
- `insurance-data-statistical-analysis.ipynb`: Main Jupyter Notebook with analysis and model training.
- `my_app.py`: Streamlit app code for predicting insurance charges.
- `insurance_model.pkl`: Saved Random Forest model.
- `scaler.pkl`: Saved StandardScaler for preprocessing `age` and `bmi`.
- `insurance.csv`: Dataset (available at [Kaggle](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset)).

## Notes
- The dataset is clean with no missing values, simplifying preprocessing.
- The Random Forest model is used in the Streamlit app due to its superior performance (R² = 0.8167).
- The app assumes `age` and `bmi` are scaled using the saved StandardScaler, matching the preprocessing in the notebook.

## Source
- Dataset: [US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset)[](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset)
- Notebook adapted from Kaggle analysis by the dataset author.