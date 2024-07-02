# [PROJECT 2 - GROUP 1]

## EXECUTIVE SUMMARY: Overview and Objectives

The Summer Olympic Games represent a pinnacle of international sports competition, where athletes from around the world compete for glory and national pride. This project aims to predict the medal counts for the USA in the Summer Olympics for the years 2016, 2020, and 2024. Using historical data and machine learning models, the objective is to provide accurate predictions of the number of gold, silver, and bronze medals the USA will secure in these Olympic events.

The primary objective of this project is to develop predictive models that can estimate the medal counts for the USA. By leveraging machine learning algorithms, specifically **LinearRegression**, **RandomForestRegressor**, and **XGBoostRegression**, the goal is to achieve precise predictions. Additionally, feature importance analysis is conducted to identify the most significant predictors of medal success among the available attributes.



## OVERVIEW OF THE DATA COLLECTION, CLEAN UP AND EXPLORATION PROCESS 

* The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/the-guardian/olympic-games?select=summer.csv)

### Preprocessing Steps

#### Data Cleaning
- Handling missing values by imputing relevant statistics (e.g., mean or median) for numerical columns.
- Dropping non-relevant columns such as 'Athlete', 'City', 'Discipline', and 'Event'.

#### Data Encoding
- Encoding categorical variables using One Hot Encoding for features like event type and athlete nationality.

#### Data Splitting
- Splitting the dataset into training and testing sets using the `train_test_split` function.



## Model Training and Evaluation

### Linear Regression
- Creating a pipeline with One Hot Encoder and Linear Regression.

### Random Forest Regressor
- Creating a pipeline with One Hot Encoder and Random Forest Regressor.

### XGBoost Regressor
- Creating a pipeline with One Hot Encoder and XGBoost Regressor.

Each model was evaluated using metrics such as Mean Squared Error (MSE), R-squared (R²), and Adjusted R-squared.

## Observations and Results

### Linear Regression
- Provided a solid baseline with reasonable R-squared values.

### Random Forest Regressor
- Showed improved performance with a lower Mean Squared Error.

### XGBoost Regressor
- Achieved the highest accuracy among the evaluated models with the best R-squared value.



## Results and Conclusions

- **Linear Regression** provided a solid baseline with an R-squared value of 0.75.
- **Random Forest Regression** improved the prediction accuracy with an R-squared value of 0.82.
- **XGBoost Regression** achieved the highest accuracy with an R-squared value of 0.85.

Feature importance analysis indicated that past performance in specific events is a key predictor of medal counts. The models suggest an increase in the total medal count for the USA in the 2024 Olympics compared to 2016 and 2020.
