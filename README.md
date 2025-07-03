# Ames Housing Price Prediction

This repository contains a data science project to predict housing prices in Ames, Iowa, using the Ames Housing Dataset. The project includes data preprocessing, model training, and evaluation.

## Getting Started

### Prerequisites

This project requires Python 3 and the following libraries:

*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   xgboost
*   lightgbm
*   joblib

You can install the necessary libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm joblib
```

### Project Structure

*   `data/`: Contains the raw and split datasets.
*   `output/`: Contains all generated outputs, including processed data, model scores, and visualizations.
*   `PROJECT_SUMMARY.md`: Provides a detailed summary of the project, the work done, and the resulting data and visualizations.

## Usage

While the scripts used to generate the project outputs have been removed, the `output/` directory contains the final results of the analysis. You can explore the generated files to see the project outcomes.

## Results

A detailed summary of the project, including the methodology and a description of all generated files, can be found in `PROJECT_SUMMARY.md`.

The key outputs of this project are:

*   Cleaned and processed data in `output/processed_train.csv` and `output/processed_test.csv`.
*   A comparison of model performance in `output/model_comparison_boxplot.png`.
*   The best performing model, LightGBM, saved as `output/LightGBM_model.joblib`.
*   Feature importance analysis of the best model in `output/feature_importance.png`.
