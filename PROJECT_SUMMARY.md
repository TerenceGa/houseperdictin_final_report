# Ames Housing Price Prediction Project

This project focuses on predicting housing prices in Ames, Iowa, using the well-known Ames Housing Dataset. This is a classic supervised regression task.

## Work Completed

The project was executed in four distinct phases:

1.  **Exploratory Data Analysis (EDA):** The initial phase involved exploring the dataset to understand its structure, identify key features, and visualize relationships between variables and the target variable, `SalePrice`.

2.  **Data Preprocessing & Feature Engineering:** The raw data was cleaned, and new features were engineered to improve model performance. This included:
    *   Applying a log transformation to the `SalePrice`.
    *   Creating combined features like `TotalSF` (total square footage) and `TotalBath` (total bathrooms).
    *   Imputing missing values using appropriate strategies.
    *   Applying one-hot encoding to categorical variables.

3.  **Model Training & Validation:** Several regression models were trained and evaluated using 5-fold cross-validation. The models included:
    *   Ridge
    *   Lasso
    *   Random Forest
    *   XGBoost
    *   LightGBM

4.  **Results & Visualization:** The performance of the models was compared, and the best-performing model (LightGBM) was analyzed further by examining its feature importances.

## Project Structure

*   `data/`: This directory contains the raw and split datasets.
    *   `AmesHousing.csv`: The original, complete dataset.
    *   `train.csv`: The training subset (80% of the original data).
    *   `test.csv`: The testing subset (20% of the original data).
*   `output/`: This directory contains all the generated outputs from the project pipeline.

## Generated Data and Graphs

The `output/` directory contains the following files:

### Processed Data

*   `processed_train.csv`: The cleaned and transformed training data, ready for modeling.
*   `processed_test.csv`: The cleaned and transformed test data.

### Model Evaluation

*   `model_scores.csv`: A table containing the Root Mean Squared Logarithmic Error (RMSLE) scores for each model from the 5-fold cross-validation.
*   `model_comparison_boxplot.png`: A box plot visually comparing the distribution of cross-validation scores for each model.

### Saved Models

*   `LightGBM_model.joblib`: The trained LightGBM model, which was the best performer.
*   `XGBoost_model.joblib`: The trained XGBoost model.
*   `RandomForest_model.joblib`: The trained Random Forest model.

### Visualizations

*   `feature_importance.png`: A bar chart showing the top 20 most important features as determined by the best-performing LightGBM model.
*   `eda/`: This subdirectory contains plots from the initial exploratory data analysis:
    *   `saleprice_histogram.png`: Histograms of the original `SalePrice` and the log-transformed `SalePrice`.
    *   `correlation_heatmap.png`: A heatmap showing the correlations between the top 10 features most correlated with `SalePrice`.
    *   `scatter_plots.png`: Scatter plots showing the relationship between `SalePrice` and key area-related features (`GrLivArea`, `TotalBsmtSF`).
