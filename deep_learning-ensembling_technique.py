
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error
from daal4py.sklearn.utils import make_daal4py_classifier
from daal4py.oneapi import sycl_context, cpu_selector
import onedal


# Loading the dataset
training_data = pd.read_csv('train_file.csv')

# Preprocessing the 'Source' column
training_data['Source'].replace('^\s+$', np.nan,regex=True,inplace=True)
training_data['Source'].replace('[-_]', '',regex=True,inplace=True)
training_data['Source'].replace('[^\x00-\x7f]', np.nan,regex=True,inplace=True)
training_data = training_data.dropna()
training_data['Source'].isna().sum()

source = list(training_data['Source'])
source_without_spaces = [(re.sub(r'[^\w]', ' ', x)).replace(' ', '') for x in source]


""" 

For Headline 

"""

headline_data = {
    'headline':list(training_data['Headline']),
    'source': source_without_spaces,
    'topic': list(training_data['Topic']),
    'facebook': list(training_data['Facebook']),
    'linkedin': list(training_data['LinkedIn']),
    'googlePlus': list(training_data['GooglePlus']),
    'target': list(training_data['SentimentHeadline'])
}

headline_df = pd.DataFrame(headline_data)

# Separate input features and target
headline_X = headline_df.drop('target', axis=1)
headline_y = headline_df['target']

# Splitting the dataset into train and test set
headline_X_train, headline_X_test, headline_y_train, headline_y_test = train_test_split(headline_X, headline_y, test_size=0.25, random_state=42)

# Preprocessing for numerical features
numerical_transformer = make_pipeline(
    StandardScaler()
)

# Preprocessing for categorical features
categorical_transformer = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)

# Preprocessing for headline feature
headline_transformer = make_pipeline(
    TfidfVectorizer()
)

# Bundle preprocessing for numerical and categorical features
headline_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['facebook', 'linkedin', 'googlePlus']),
        ('cat', categorical_transformer, ['source', 'topic']),
        ('headline', headline_transformer, 'headline')
    ])

# Defining the base model
headline_base_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Defining the stacking model
headline_stacking_model = StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=10, random_state=42))
    ],
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)
)

# Defining the full pipeline with preprocessing and the stacked model
full_headline_pipeline = Pipeline(steps=[
    ('preprocessor', headline_preprocessor),
    ('stacked_model', headline_stacking_model)
])

# Fit the pipeline
with sycl_context(cpu_selector()):
    full_headline_pipeline.fit(headline_X_train, headline_y_train)

# Predict with the pipeline
with sycl_context(cpu_selector()):
    y_headline_pred = full_headline_pipeline.predict(headline_X_test)

# Calculate the mean absolute error
headline_mae = mean_absolute_error(headline_y_test, y_headline_pred)

print("Mean Absolute Error for Headline:", headline_mae)


""" 

For Title 

"""


title_data = {
    'title':list(training_data['Title']),
    'source': source_without_spaces,
    'topic': list(training_data['Topic']),
    'facebook': list(training_data['Facebook']),
    'linkedin': list(training_data['LinkedIn']),
    'googlePlus': list(training_data['GooglePlus']),
    'target': list(training_data['SentimentTitle'])
}

title_df = pd.DataFrame(title_data)

# Separate input features and target
title_X = title_df.drop('target', axis=1)
title_y = title_df['target']

# Splitting the dataset into train and test set
title_X_train, title_X_test, title_y_train, title_y_test = train_test_split(title_X, title_y, test_size=0.25, random_state=42)

# Preprocessing for numerical features
numerical_transformer = make_pipeline(
    StandardScaler()
)

# Preprocessing for categorical features
categorical_transformer = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)

# Preprocessing for title feature
title_transformer = make_pipeline(
    TfidfVectorizer()
)

# Bundle preprocessing for numerical and categorical features
title_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['facebook', 'linkedin', 'googlePlus']),
        ('cat', categorical_transformer, ['source', 'topic']),
        ('title', title_transformer, 'title')
    ])

# Defining the base model
title_base_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Defining the stacking model
title_stacking_model = StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=10, random_state=42))
    ],
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)
)

# Defining the full pipeline with preprocessing and the stacked model
full_title_pipeline = Pipeline(steps=[
    ('preprocessor', title_preprocessor),
    ('stacked_model', title_stacking_model)
])

# Fit the pipeline
with sycl_context(cpu_selector()):
    full_title_pipeline.fit(title_X_train, title_y_train)

# Predict with the pipeline
with sycl_context(cpu_selector()):
    y_title_pred = full_title_pipeline.predict(title_X_test)

# Calculate the mean absolute error
title_mae = mean_absolute_error(title_y_test, y_title_pred)

print("Mean Absolute Error for Title:", title_mae)
