# [PROJECT 2 - GROUP 7]

## EXECUTIVE SUMMARY: Overview and Objectives

Stroke is a significant global health concern, ranking as the second leading cause of death worldwide, according to the World Health Organization (WHO). Predicting the likelihood of stroke occurrence in patients can aid in preventive healthcare measures. This project utilizes a dataset comprising various demographic and health-related parameters to predict the likelihood of stroke occurrence in individuals. Parameters such as age, gender, presence of hypertension, heart disease, marital status, occupation, residence type, average glucose level, BMI, and smoking status are considered for prediction.

The primary objective of this project is to develop predictive models to identify individuals at risk of stroke based on their demographic and health-related attributes. By leveraging machine learning algorithms, specifically logistic regression and random forest classification, the aim is to achieve accurate predictions. Additionally, feature importance analysis is conducted to identify the most significant predictors of stroke risk among the available attributes.

## OVERVIEW OF THE DATA COLLECTION, CLEAN UP AND EXPLORATION PROCESS 

* The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)
* Upon loading the dataset into a Pandas dataframe, several preprocessing steps are undertaken.
* Initial data exploration revealed missing values in the BMI column, which were imputed using the mean BMI value.
* Age, a continuous variable, was binned to minimize unique features.

## APPROACH
At a high level the steps taken were data preprocessing, model development, and evaluation. The code involves various libraries and modules including Pandas, Scikit-learn, Matplotlib, and imbalanced-learn. Key functionalities include data loading, preprocessing steps such as missing value imputation and encoding, model training using logistic regression and random forest classifiers, model evaluation, and feature importance analysis. Standard scaling is applied to the input data to ensure optimal model performance.

Steps
* The dataset was first encoded, then scaled and lastly split into training and testing sets.
* Both logistic regression and random forest classification models were trained on the data.
* The RandomOverSampler and RandomUnderSampler techniques were employed to handle class imbalance.
* The performance of each model was evaluated using metrics such as accuracy, balanced accuracy, and confusion matrices.
* Classification reports were generated to assess model performance for both stroke and no-stroke classes.

## FUTURE DEVELOPMENT AND ADDITIONAL QUESTIONS
1. Feature engineering and selection techniques can be applied to identify and include only the most relevant features, reducing noise and improving the models' ability to capture underlying patterns.
2. Further efforts, such as data augmentation techniques and model optimization, or even a more robust data set are warranted to enhance the models' performance and enable more accurate risk assessment.
3. Would be great to test different theories and order of steps to determine if results vary 

## RESULTS AND CONCLUSIONS
The dataset used in this project is unbalanced, so we employed resampling techniques such as oversampling the minority class (stroke occurrences) and undersampling the majority class (non-stroke occurrences) to help balance the dataset. Despite the inherent challenges posed by an unbalanced dataset, the developed models still provide valuable insights. Logistic regression achieves a training data score of [ score] and a testing data score of [ score]. Similarly, the random forest classifier achieves a training data score of [score] and a testing data score of [score]. These scores ultimately reflect optimal performance.

Feature importance analysis remains instrumental in understanding the key factors influencing stroke risk, by identifying the most significant predictors, healthcare professionals can prioritize interventions and preventive measures for individuals at higher risk.
