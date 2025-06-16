# Machine Learning for Solar Power Prediction

## Project Overview

This project develops and evaluates a series of machine learning models to predict the power generation of solar power plants based on time-series weather data. The goal is to determine the most effective modeling approach for this regression task by comparing a powerful tree-based model against a deep learning architecture.

The workflow covers the entire data science pipeline: data curation and aggregation from multiple sources, time-based feature engineering, exploratory data analysis (EDA) to understand the physical relationships in the data, and a comparative evaluation of an XGBoost Regressor and a Neural Network.

## Key Findings

1.  **Primary Driver Identified:** Exploratory Data Analysis confirmed a strong, positive, and mostly linear relationship between **Solar Irradiation** and **AC Power Output**. This identifies irradiation as the most important predictive feature.
2.  **Temperature's Impact:** The analysis also revealed a clear "temperature derating" effect, where the solar panels' power output begins to decrease at very high module temperatures, even with high irradiation. This confirms that `MODULE_TEMPERATURE` is a crucial secondary feature.
3.  **Superior Baseline Model:** A baseline model using an **XGBoost Regressor** achieved excellent performance (**R-squared of ~0.96**), setting a very high benchmark for this prediction task.
4.  **Comparative Model Analysis:** A more complex Neural Network was built and evaluated but **did not outperform the XGBoost model**. This provides a key conclusion: for this well-structured, tabular regression problem, the optimized tree-based model was the more effective solution.

## Dataset

* **Solar Power Generation Data:** The dataset, sourced from Kaggle, contains 34 days of 15-minute time-series data from two separate solar power plants. The data is provided in four CSV files, which were aggregated and merged for this project.

## Methodology & Tools

1.  **Data Curation:** Raw data was loaded, and `DATE_TIME` columns were converted to the proper format. Inverter-level power generation data was aggregated to the plant level for each timestamp to match the granularity of the weather data. The final clean dataset was saved to `solar_curated_data.pkl`.
2.  **Feature Engineering:** Time-based features—`MONTH`, `DAY_OF_YEAR`, `HOUR`, and `MINUTE`—were extracted from the `DATE_TIME` column to allow the models to learn daily and seasonal patterns.
3.  **Machine Learning Preparation:** The data was separated into features (X) and a target (`AC_POWER`). After performing a train-test split, all features were scaled using `scikit-learn`'s `StandardScaler`.
4.  **Modeling & Evaluation:** An **XGBoost Regressor** was trained as a baseline and compared against a **Feed-Forward Neural Network** built with `TensorFlow/Keras`. Both were evaluated using R-squared (<span class="math-inline">R^2</span>), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Model Performance Comparison

The final evaluation showed a clear performance difference between the two models.

| Metric | XGBoost (Baseline) | Neural Network |
| :--- | :--- | :--- |
| **R-squared (<span class="math-inline">R^2</span>)** | **0.9578** | 0.9403 |
| **MAE (kW)** | **587.20** | 846.80 |
| **RMSE (kW)** | **1562.87**| 1859.72 |

**Conclusion:** The **XGBoost model is the clear winner**, outperforming the neural network on every key metric.

---

## Repository Contents

* **
