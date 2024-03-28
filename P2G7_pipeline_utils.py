import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def preprocess_healthcare_data(healthcare_df):
    """
    Written for healthcare data; will drop null values and 
    split into training and testing sets. Uses stroke
    as the target column.
    """
    raw_num_df_rows = len(healthcare_df)
    healthcare_df = healthcare_df.dropna()
    remaining_num_df_rows = len(healthcare_df)
    percent_na = (
        (raw_num_df_rows - remaining_num_df_rows) / raw_num_df_rows * 100
    )
    print(f"Dropped {round(percent_na,2)}% rows")
    X = healthcare_df.drop(columns='stroke')
    # y = healthcare_df['stroke'].values.reshape(-1, 1)
    y = healthcare_df['stroke']
    return train_test_split(X, y)

def preprocess_healthcare_data_keep_na(healthcare_df):
    """
    Written for healthcare data; will split into training
    and testing sets. Uses stroke as the target column.
    """
    X = healthcare_df.drop(columns='stroke')
    # y = healthcare_df['stroke'].values.reshape(-1, 1)
    y = healthcare_df['stroke']
    return train_test_split(X, y)

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, model):
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"Adjusted R-squared: {r2_adj(X_test, y_test, model)}")

    return r2_adj(X_test, y_test, model)

def get_best_pipeline(pipeline, pipeline2, healthcare_df):
    """
    Accepts two pipelines and healthcare data.
    Uses two different preprocessing functions to 
    split the data for training the different 
    pipelines, then evaluates which pipeline performs
    best.
    """
    # Apply the preprocess_healthcare_data step
    X_train, X_test, y_train, y_test = preprocess_healthcare_data(healthcare_df)

    # Fit the first pipeline
    pipeline.fit(X_train, y_train)

    print("Testing dropped NAs")
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the first pipeline
    p1_adj_r2 = check_metrics(X_test, y_test, pipeline)

    # Apply the preprocess_healthcare_data_keep_na step
    X_train, X_test, y_train, y_test = preprocess_healthcare_data_keep_na(healthcare_df)

    # Fit the second pipeline
    pipeline2.fit(X_train, y_train)

    print("Testing no dropped data")
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the second pipeline
    p2_adj_r2 = check_metrics(X_test, y_test, pipeline2)

    # Compare the adjusted r-squared for each pipeline and 
    # return the best model
    if p2_adj_r2 > p1_adj_r2:
        print("Returning no dropped data pipeline")
        return pipeline2
    else:
        print("Returning dropped NAs pipeline")
        return pipeline
    
def healthcare_model_generator(healthcare_df):
    """
    Defines a series of steps that will preprocess data,
    split data, and train a model for predicting stroke 
    using linear regression. It will return the trained model
    and print the mean squared error, r-squared, and adjusted
    r-squared scores.
    """
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function
    steps = [("One hot encode", OneHotEncoder(handle_unknown="ignore")), 
             ("Scale", StandardScaler(with_mean=False)), 
             ("Logistic Regression", LogisticRegression(random_state=7, max_iter=120))] 

    # Create a pipeline object
    pipeline = Pipeline(steps)

    # Create a second pipeline object
    pipeline2 = Pipeline(steps)

    # Get the best pipeline
    pipeline = get_best_pipeline(pipeline, pipeline2, healthcare_df)

    # Return the trained model
    return pipeline

def train_test_split_healthcare(df_encoded):
    X = df_encoded.drop(columns='stroke')
    y = df_encoded['stroke']
    return train_test_split(X, y, random_state=13)

def fill_bmi_na(df_copy):
    bmi_mean = df_copy['bmi'].mean()
    df_copy['bmi'] = df_copy['bmi'].fillna(bmi_mean)
    return df_copy

def bin_age(df_bmi_filled):
    bin_labels = [1, 2, 3, 4]
    df_bmi_filled['bin_age'] = pd.qcut(df_bmi_filled['age'], q=4, labels=bin_labels)
    return df_bmi_filled

def drop_id_age(df_bmi_filled_age_bin):
    df_bmi_filled_age_bin = df_bmi_filled_age_bin.drop(columns=['id', 'age'])
    return df_bmi_filled_age_bin

def delete_gender_other(df_drop_id_age):
    df_drop_id_age = df_drop_id_age[df_drop_id_age['gender'] != 'Other']
    return df_drop_id_age
    
def change_smoking_status_Unknown(df_encoded):
    df_encoded['x0_Unknown'] = df_encoded['x0_Unknown'].replace({1.0: -1.0})
    return df_encoded

#Functions for building and training encoders
def build_smoking_status_encoder(df_final):
    smoking_status_encoder = OneHotEncoder(max_categories=None, handle_unknown='error', sparse_output=False)
    # Train the encoder
    smoking_status_encoder.fit(df_final['smoking_status'].values.reshape(-1, 1))
    return {'column': 'smoking_status',
            'multi_col_output': True,
            'encoder': smoking_status_encoder}

def build_work_type_encoder(df_final):
    work_type_encoder = OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False)

    # Train the encoder
    work_type_encoder.fit(df_final['work_type'].values.reshape(-1, 1))
    return {'column': 'work_type',
            'multi_col_output': True,
            'encoder': work_type_encoder}

def build_Residence_type_encoder(df_final):
    Residence_type_encoder = OrdinalEncoder(categories=[['Urban', 'Rural']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    Residence_type_encoder.fit(df_final['Residence_type'].values.reshape(-1, 1))
    return {'column': 'Residence_type',
            'multi_col_output': False,
            'encoder': Residence_type_encoder}

def build_gender_encoder(df_final):
    gender_encoder = OrdinalEncoder(categories=[['Male', 'Female']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    gender_encoder.fit(df_final['gender'].values.reshape(-1, 1))
    return {'column': 'gender',
            'multi_col_output': False,
            'encoder': gender_encoder}

def build_ever_married_encoder(df_final):
    ever_married_encoder = OrdinalEncoder(categories=[['No', 'Yes']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    ever_married_encoder.fit(df_final['ever_married'].values.reshape(-1, 1))
    return {'column': 'ever_married',
            'multi_col_output': False,
            'encoder': ever_married_encoder}

def build_encoders(df_final):
    encoder_functions = [build_smoking_status_encoder, 
                         build_work_type_encoder, 
                         build_Residence_type_encoder, 
                         build_gender_encoder,
                         build_ever_married_encoder
                        ]
    return [encoder_function(df_final) for encoder_function in encoder_functions]


# Encoding all categorical variables
def encode_categorical(df_final, encoders):
    # Separate numeric columns
    dfs = [df_final.select_dtypes(include='number').reset_index(drop=True)]

    single_col_encoders = []
    for encoder_dict in encoders:
        encoder = encoder_dict['encoder']
        column = encoder_dict['column']
        multi_col = encoder_dict['multi_col_output']
        if not multi_col:
            single_col_encoders.append(encoder_dict)
        else:
            dfs.append(pd.DataFrame(encoder.transform(df_final[column].values.reshape(-1, 1)), columns=encoder.get_feature_names_out()))
    
    df_final_encoded = pd.concat(dfs, axis=1)

    for encoder_dict in single_col_encoders:
        encoder = encoder_dict['encoder']
        column = encoder_dict['column']
        multi_col = encoder_dict['multi_col_output']
        df_final_encoded[column] = encoder.transform(df_final[column].values.reshape(-1, 1))

    return df_final_encoded

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")

"""
def build_target_encoder(y):
    encode_y = OneHotEncoder(drop='first', sparse_output=False)
    encode_y.fit(y)
    return encode_y

def encode_target(y, encode_y):
    
    return np.ravel(encode_y.transform(y))
"""    
