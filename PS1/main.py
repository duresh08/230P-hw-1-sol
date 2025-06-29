import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def rolling_features(df, window_list=[20]):
    '''
    Adds rolling mean, std, min, max, range, and returns for each numeric feature per firm.
    '''
    # Identify macro features to exclude from rolling calculations
    macro1_features = [col for col in df.columns if col.startswith('macro1')]
    
    # Select all numeric columns excluding identifiers and macro1
    columns = [col for col in df.columns if col not in ['firm_id', 'date', 'ret'] + macro1_features]

    # Iterate over each unique firm
    for firm_id in tqdm(df['firm_id'].unique(), desc='Rolling features per firm_id'):
        for col in columns:
            firm_mask = df['firm_id'] == firm_id
            for window in window_list:
                # Rolling average
                df.loc[firm_mask, f'{col}_rolling_avg_{window}'] = df.loc[firm_mask, col].rolling(window).mean()

                # Rolling standard deviation
                df.loc[firm_mask, f'{col}_rolling_std_{window}'] = df.loc[firm_mask, col].rolling(window).std()

                # Rolling max
                df.loc[firm_mask, f'{col}_rolling_max_{window}'] = df.loc[firm_mask, col].rolling(window).max()

                # Rolling min
                df.loc[firm_mask, f'{col}_rolling_min_{window}'] = df.loc[firm_mask, col].rolling(window).min()

                # High/low ratio (max / min)
                df.loc[firm_mask, f'{col}_rolling_high/low_{window}'] = (
                    df.loc[firm_mask, f'{col}_rolling_max_{window}'] /
                    df.loc[firm_mask, f'{col}_rolling_min_{window}']
                )

                # Range = max - min
                df.loc[firm_mask, f'{col}_rolling_range_{window}'] = (
                    df.loc[firm_mask, f'{col}_rolling_max_{window}'] -
                    df.loc[firm_mask, f'{col}_rolling_min_{window}']
                )

                # Window-based return (current / lag - 1)
                df.loc[firm_mask, f'{col}_rolling_ret_{window}'] = (
                    df.loc[firm_mask, col] / df.loc[firm_mask, col].shift(window - 1) - 1
                )
    return df

def feature_transformations(df):
    '''
    Adds lag, squared, cubed, and log-return transformations per firm.
    '''
    # Exclude macro1 from transformations
    macro1_features = [col for col in df.columns if col.startswith('macro1')]
    
    # Select columns to transform
    columns = [col for col in df.columns if col not in ['firm_id', 'date', 'ret'] + macro1_features]

    for firm_id in tqdm(df['firm_id'].unique(), desc='Feature transformations per firm_id'):
        firm_mask = df['firm_id'] == firm_id

        for col in columns:
            # Lag-1 feature
            df.loc[firm_mask, f'{col}_lag_1'] = df.loc[firm_mask, col].shift(1)

            # Polynomial features
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_cubed'] = df[col] ** 3

        # Log return for price (requires lag column to be generated first)
        df.loc[firm_mask, f'price_log_ret'] = (
            np.log(df.loc[firm_mask, 'price'] / df.loc[firm_mask, 'price_lag_1'])
        )

    return df

def feature_one_hot_encoding(df):
    '''
    One-hot encodes the 'macro1' column.
    '''
    print('One-hot encoding categorical feature: macro1')

    # Create dummy columns
    dummies = pd.get_dummies(df['macro1'], prefix='macro1', drop_first=False)

    # Drop original and concatenate dummies
    df = pd.concat([df.drop(columns=['macro1']), dummies], axis=1)
    return df

def firm_id_one_hot_encoding(df):
    '''
    One-hot encodes the 'firm_id' column.
    '''
    print('One-hot encoding firm_id')

    # Create dummy columns
    dummies = pd.get_dummies(df['firm_id'], prefix='firm_id', drop_first=False)

    # Drop original and concatenate dummies
    df = pd.concat([df.drop(columns=['firm_id']), dummies], axis=1)
    return df

def feature_interactions(df):
    '''
    Creates interaction terms between firm variables, macro variables, and price.
    '''
    print('Creating feature interactions')

    # Firm-to-firm interactions
    df['firm1*firm2'] = df['firm1'] * df['firm2']
    df['firm2*firm3'] = df['firm2'] * df['firm3']
    df['firm1*firm3'] = df['firm1'] * df['firm3']

    # Firm-to-macro interactions
    df['firm1*macro2'] = df['firm1'] * df['macro2']
    df['firm2*macro2'] = df['firm2'] * df['macro2']
    df['firm3*macro2'] = df['firm3'] * df['macro2']

    # Firm-to-price interactions
    df['firm1*price'] = df['firm1'] * df['price']
    df['firm2*price'] = df['firm2'] * df['price']
    df['firm3*price'] = df['firm3'] * df['price']

    # Macro-to-price interaction
    df['macro2*price'] = df['macro2'] * df['price']

    # Price ratio features
    df['price/macro2'] = df['price'] / df['macro2']
    df['price/firm1'] = df['price'] / df['firm1']
    df['price/firm2'] = df['price'] / df['firm2']
    df['price/firm3'] = df['price'] / df['firm3']

    return df

def full_feature_engineering_pipeline(df):
    '''
    Applies all feature engineering steps using method chaining via .pipe.
    '''
    return (df.pipe(feature_interactions)
              .pipe(feature_one_hot_encoding)
              .pipe(feature_transformations)
              .pipe(rolling_features)
              .pipe(firm_id_one_hot_encoding))
    
# loading models
elastic_net_model = joblib.load('./elasticnet_model.pkl')
xgboost_model = joblib.load('./xgboost_model.pkl')

# get scaler and columns to standardize from pickle file
scaled_cols = xgboost_model['scaled_cols']
scaler = xgboost_model['scaler']

# create model map for fat inferencing
model_map = {'xgb': xgboost_model, 'elasticnet': elastic_net_model}

# read the unseen data sample
test = pd.read_parquet('./new_data.parquet')
test_dates = sorted(list(test['date'].unique()))

# concatenate last 25 days of training data along with unseen test data
training_data = pd.read_parquet('./training_data.parquet').drop(columns = ['ret'], errors='ignore')
train_data_with_features = pd.read_parquet('./train+features.parquet').drop(columns = ['ret'], errors='ignore')

test = pd.concat([training_data[training_data['date'].isin(sorted(training_data['date'].unique())[-25:])], test])

# engineer features for test data
test = full_feature_engineering_pipeline(test)
test = test[test['date'].isin(test_dates)].reset_index(drop = True)

# scale the entire dataframe based on scaler created during pickling on training data only
test[scaled_cols] = scaler.transform(test[scaled_cols])

# in case all categorical macro1 trough contraction present in test data
missing_cols = [i for i in list(set(train_data_with_features.columns) ^ set(test.columns)) if i != 'ret']
test[missing_cols] = False

# drop date from the test dataframe
test = test.drop(columns = ['date'])
test = test.loc[:, [i for i in train_data_with_features.columns if i not in ['date', 'ret']]]

# get predictions for elasticnet and xgboost models
xgb_pred = xgboost_model['model'].predict(test)
elasticnet_pred = elastic_net_model['model'].predict(test)

# saving the sample predictions
print('saving sample predictions')
pd.DataFrame(xgb_pred).to_csv('./xgboost_sample_submission.csv', index = False)
pd.DataFrame(elasticnet_pred).to_csv('./elasticnet_sample_submission.csv', index = False)