# [PROJECT 2 - GROUP 7]

## EXECUTIVE SUMMARY: Overview and Objectives

Stroke is a significant global health concern, ranking as the second leading cause of death worldwide, according to the World Health Organization (WHO). Predicting the likelihood of stroke occurrence in patients can aid in preventive healthcare measures. This project utilizes a dataset comprising various demographic and health-related parameters to predict the likelihood of stroke occurrence in individuals. Parameters such as age, gender, presence of hypertension, heart disease, marital status, occupation, residence type, average glucose level, BMI, and smoking status are considered for prediction.

The primary objective of this project is to develop predictive models to identify individuals at risk of stroke based on their demographic and health-related attributes. By leveraging machine learning algorithms, specifically logistic regression and random forest classification, the aim is to achieve accurate predictions. Additionally, feature importance analysis is conducted to identify the most significant predictors of stroke risk among the available attributes.

## OVERVIEW OF THE DATA COLLECTION, CLEAN UP AND EXPLORATION PROCESS 

* The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)
* At a high level the steps taken were data preprocessing, model development, and evaluation. The code involves various libraries and modules including Pandas, Scikit-learn, Matplotlib, and imbalanced-learn. Key functionalities include data loading, preprocessing steps such as missing value imputation and encoding, model training using logistic regression and random forest classifiers, model evaluation, and feature importance analysis. Standard scaling is applied to the input data to ensure optimal model performance.
* Preprocessing steps such as the below were undertaken:
      - Imputting the mean BMI value for missing values in the BMI column.
      - Age, a continuous variable, was binned to minimize unique features.
* The dataset was first encoded using the One Hot Encoder method for Smoking status and Work Type; and the Ordinal Encoder for Residence Type, Gender, and Ever Married features.  
* Subsequently the dataframe was scaled taking both Numerical and Categorical values. 
* Lastly, we applied split into training and testing sets.
* Both logistic regression and random forest classification models were trained on the data using varied methodologies
* We conducted a Base Model analysis for the following models:
       - Support Vector Machine (SVM)
      - K Nearest Neighbors (KNN)
      - Extra Trees Classifier
      - Gradient Boosting Classifier
      - Ada Boost Classifier
* The performance of each model was evaluated using metrics such as accuracy, balanced accuracy, and confusion matrices.
* Classification reports were generated to assess model performance for both stroke and no-stroke classes.
* The RandomOverSampler and RandomUnderSampler techniques were employed to handle class imbalance and these were our observations and results:
    1. The Accuracy from Over Sampled data is 98%, which is very high. This might be due to over-fitting.
    2. The Accuracy from UnderSampled data is 77% which is more in-line with healthcare data
    3. Despite the above result being a promising one, we realized the False Negative value of 16 from the Confusion Matrix was still high and we wanted to check if varying the number of test data (using the same model) would decrease the False Negative count.
       a) We created a new Test data set by randomly selecting 1000 rows from the original dataset.
       b) The selection included 941 No Stroke instances and 59 Stroke instances.
       c) Running this data through the previous Under Sample model gave us 72% accuracy.
       d) However, the False Negative value decreased from 16 to 6 for this bigger test data set.
       e) Ratio of False Positive to False Negative is much higher with this data set.
       * We have more confidence in the prediction of this model!! 

![image](https://github.com/mvenegas011/Project2_Group7/assets/33967792/460746b8-4d1b-488b-9c27-9c6b8321ca0f)


## RESULTS AND CONCLUSIONS
1. Over Sampled model has more Precision but less Sensitivity.
2. Under Sample model is less Precise but more Sensitive i.e. higher ratio of False Positives to False Negative
3. In healthcare data, we want more Sensitivity compared to Precision.
4. For data set with unbalanced target, tree based algorithms and boosting algorithms are ideal models.
5. Amongst the Decision Tree models, Random Forest and Extra Trees Classifier has slightly higher Balanced Accuracy score. 
