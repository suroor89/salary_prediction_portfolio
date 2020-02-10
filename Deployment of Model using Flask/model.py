# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import pickle

def load_data(file):
    return pd.read_csv(file)
train_feature_df = load_data('train_features.csv')
train_target_df = load_data('train_salaries.csv')
test_feature_df = load_data('test_features.csv')

#Consolidate training data: Performing inner join and getting uniq rows from both dataframe
def consolidate_data(df1, df2, key=None, left_index=False, right_index=False):
    return pd.merge(left=df1, right=df2, how='inner', on=key, left_index=left_index, right_index=right_index) 

raw_train_df = consolidate_data(train_feature_df, train_target_df, key='jobId') 


def clean_data(raw_df):
    clean_df = raw_df.drop_duplicates(subset='jobId')
    clean_df = clean_df[clean_df.salary > 0]
    return clean_df

'''Shuffle, and reindex training data -- shuffling improves cross-validation accuracy'''
clean_train_df = shuffle(clean_data(raw_train_df)).reset_index()

def merge_cat_num_data(df1,df2):# cat_vars=None, num_vars=None):
    train_df = pd.merge(df1, df2)
    return train_df 
merged_data = merge_cat_num_data(clean_train_df,train_target_df)

# Remove jobId and companyId
clean_train_df.drop('index', axis=1, inplace=True)
clean_train_df.drop('jobId', axis=1, inplace=True)
clean_train_df.drop('companyId', axis=1, inplace=True)



# One-hot encode categorical data in train_data dataset
oenHot_data= pd.get_dummies(clean_train_df[['jobType','degree','major','industry']], drop_first= True)


clean_train_df.drop('jobType', axis=1, inplace=True)
clean_train_df.drop('degree', axis=1, inplace=True)
clean_train_df.drop('major', axis=1, inplace=True)
clean_train_df.drop('industry', axis=1, inplace=True)

final_data = pd.concat([clean_train_df, oenHot_data], axis=1)#,ignore_index=False)

#print(final_data.head())
#X = final_data[final_data.loc[ : , final_data.columns != 'salary'].columns]
X = final_data.iloc[:, 0:2]
y = final_data['salary']



#create Linear Regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 5]]))





