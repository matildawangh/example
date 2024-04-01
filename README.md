# [PROJECT 2 - GROUP 7]

## EXECUTIVE SUMMARY: Overview and Objectives

Stroke is a significant global health concern, ranking as the second leading cause of death worldwide, according to the World Health Organization (WHO). Predicting the likelihood of stroke occurrence in patients can aid in preventive healthcare measures. This project utilizes a dataset comprising various demographic and health-related parameters to predict the likelihood of stroke occurrence in individuals. Parameters such as age, gender, presence of hypertension, heart disease, marital status, occupation, residence type, average glucose level, BMI, and smoking status are considered for prediction.

The primary objective of this project is to develop predictive models to identify individuals at risk of stroke based on their demographic and health-related attributes. By leveraging machine learning algorithms, specifically logistic regression and random forest classification, the aim is to achieve accurate predictions. Additionally, feature importance analysis is conducted to identify the most significant predictors of stroke risk among the available attributes.

## OVERVIEW OF THE DATA COLLECTION, CLEAN UP AND EXPLORATION PROCESS 

The dataset used for this project is sourced from Kaggle. Upon loading the dataset into a Pandas dataframe, several preprocessing steps are undertaken. Missing values in the 'bmi' column are filled with the mean BMI value. Age data is binned to reduce uniqueness, and irrelevant features such as 'id' is dropped, the 'age' is 'binned' to allow for easier categorization. Categorical variables are encoded using one-hot encoding, and 'Unknown' values in the 'smoking_status' column are handled appropriately. The dataset is split into training and testing sets for model development and evaluation.

https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

DETAILS ON CLEAN UP PROCESS:
* Bin the 'age'. Do histogram first to see distribution. And then encode it.
* Proposed bin size
* 0 -18
* 19 - 30
* 30 - 50
* 50 - 65
* 65 - 72
* 72 - 80
* Some nulls in BMI (two datasets 1. null dropped, 2. null replaced with some value - mean is good 28.1)
* 1544 Unknown Smokers. What do we do? Can we put negative one. 

## APPROACH
Python code is built and utilized for data preprocessing, model development, and evaluation. The code involves various libraries and modules including Pandas, Scikit-learn, Matplotlib, and imbalanced-learn. Key functionalities include data loading, preprocessing steps such as missing value imputation and encoding, model training using logistic regression and random forest classifiers, model evaluation, and feature importance analysis. Standard scaling is applied to the input data to ensure optimal model performance.

## FUTURE DEVELOPMENT AND ADDITIONAL QUESTIONS
1. The dataset used in this project is unbalanced, so with more time we could search for a more complete dataset
2. Employ resampling techniques such as oversampling the minority class (stroke occurrences) or undersampling the majority class (non-stroke occurrences) can help balance the dataset
3. Feature engineering and selection techniques can be applied to identify and include only the most relevant features, reducing noise and improving the models' ability to capture underlying patterns.   

By implementing these strategies, the models can be refined and optimized to better predict stroke occurrences and assist healthcare professionals in making informed decisions regarding preventive measures and interventions.

## RESULTS AND CONCLUSIONS
Despite the inherent challenges posed by an unbalanced dataset and lower performance indicated by confusion matrix results, the developed models still provide valuable insights. Logistic regression achieves a training data score of [ score] and a testing data score of [ score]. Similarly, the random forest classifier achieves a training data score of [score] and a testing data score of [score]. Although these scores may not reflect optimal performance, they serve as a foundation for further refinement and improvement.

Feature importance analysis remains instrumental in understanding the key factors influencing stroke risk, despite the dataset's imbalance. By identifying the most significant predictors, healthcare professionals can prioritize interventions and preventive measures for individuals at higher risk.

In conclusion, while acknowledging the limitations imposed by data imbalance and lower confusion matrix results, the developed models still offer valuable insights into stroke prediction. Further efforts, such as data augmentation techniques and model optimization, are warranted to enhance the models' performance and enable more accurate risk assessment. These endeavors are crucial in advancing preventive healthcare strategies aimed at reducing the burden of stroke-related morbidity and mortality.
