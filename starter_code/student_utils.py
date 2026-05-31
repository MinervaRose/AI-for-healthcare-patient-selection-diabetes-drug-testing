import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

### Question 3: Reduce dimensionality of NDC codes
def reduce_dimension_ndc(df, ndc_df):
    '''
    Merges NDC code info into the main dataframe to add generic drug names.
    '''
    ndc_df = ndc_df.rename(columns={
        'NDC_Code': 'ndc_code',
        'Non-proprietary Name': 'generic_drug_name'
    })
    df = df.merge(ndc_df[['ndc_code', 'generic_drug_name']], on='ndc_code', how='left')
    return df


### Question 4: Select first encounter for each patient
def select_first_encounter(df):
    '''
    Sorts by encounter_id and selects the first hospital encounter per patient.
    '''
    df_sorted = df.sort_values(by=['patient_nbr', 'encounter_id'])
    first_encounter_df = df_sorted.drop_duplicates(subset='patient_nbr', keep='first')
    return first_encounter_df


### Question 6: Patient-based train/val/test split
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    Splits the dataset by patient to avoid data leakage between sets.
    '''
    unique_patients = df[patient_key].unique()
    np.random.shuffle(unique_patients)

    total = len(unique_patients)
    train_cut = int(0.6 * total)
    val_cut = int(0.8 * total)

    train_patients = unique_patients[:train_cut]
    val_patients = unique_patients[train_cut:val_cut]
    test_patients = unique_patients[val_cut:]

    train = df[df[patient_key].isin(train_patients)].reset_index(drop=True)
    val = df[df[patient_key].isin(val_patients)].reset_index(drop=True)
    test = df[df[patient_key].isin(test_patients)].reset_index(drop=True)

    return train, val, test


### Question 7: TensorFlow categorical feature columns
def create_tf_categorical_feature_cols(categorical_col_list, vocab_dir='./diabetes_vocab/'):
    '''
    Creates TensorFlow indicator columns from vocab files.
    '''
    output_tf_list = []
    for col in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir, col + "_vocab.txt")
        cat_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=col, vocabulary_file=vocab_file_path, num_oov_buckets=1
        )
        indicator_col = tf.feature_column.indicator_column(cat_column)
        output_tf_list.append(indicator_col)
    return output_tf_list


### Question 8: TensorFlow numeric feature columns
def normalize_numeric_with_zscore(col, mean, std):
    return (col - mean) / std

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    return tf.feature_column.numeric_column(
        key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64
    )


### Question 9: Extract mean and std from TF Probability predictions
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object (Normal distribution)
    '''
    m = diabetes_yhat.mean().numpy().flatten()
    s = diabetes_yhat.stddev().numpy().flatten()
    return m, s



### Question 10: Binary prediction based on regression mean
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe
    col: string, column name to use for prediction thresholding
    return: np.array of 1s and 0s
    '''
    return df[col].apply(lambda x: 1 if 5 <= x <= 7 else 0).values


