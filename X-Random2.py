import re
import math
import time
import random
from random import choice
import numpy as np
import pandas as pd
import collections
from sklearn.linear_model import LogisticRegression

%matplotlib inline


def Data_PreProcessing(train, actions, test):
    
    actions = actions.fillna(0)
    test = test.fillna('missing')
    actions = actions.drop(['plan_id', 'spark_plan_name', 'unique_tag', 'tag', 'components'], axis=1)
    test = test.drop(['user_id'], axis=1)
    for i in range(len(train)//2, len(train)):
        train.loc[i, 'REWARD'] = 0
    
    train, test, actions = OHE_Data(train, actions, test)
    
    return train, actions, test

def OHE_Function(column, values, train):                                 #One-hot-encoding of data
    
    c = []
    for val in values:
        val = str(val).upper()+str('_')+str(column)
        c.append(val)
    
    dummy = pd.DataFrame(np.zeros((len(train), len(values))), columns=c)
    
    for i in range(len(train)):
        val = train.loc[i, column]
        if val == 'missing':
            continue
        else:
            col = str(val).upper()+str('_')+str(column)
            dummy.loc[i, col] = 1
        
    return dummy

def OHE_Data(train, actions, test):                                      #Which data gets one-hot-encoded
    
    train_ohe = pd.DataFrame([i for i in range(len(train))], columns=['Index'])
    for column in train.columns:
        if type(train[column][0]) == np.int64:
            train_ohe=train_ohe.join(train[column])
            continue
        values = list(set(train[column].tolist()))
        df = OHE_Function(column, values, train)
        train_ohe=train_ohe.join(df)
    del train_ohe['Index']
    
    actions_ohe = pd.DataFrame([i for i in range(len(actions))], columns=['Index'])
    for column in actions.columns:
        if type(actions[column][0]) == np.int64 or column in ['CREDIT', 'TAX','LMF', 'INSURANCE', 'GOLD']:
            actions_ohe=actions_ohe.join(actions[column])
            continue
        values = list(set(actions[column].tolist()))
        df = OHE_Function(column, values, actions)
        actions_ohe=actions_ohe.join(df)
    del actions_ohe['Index']
    
    test_ohe = pd.DataFrame([i for i in range(len(test))], columns=['Index'])
    for column in test.columns:
        if type(test[column][0]) == np.int64:
            test_ohe=test_ohe.join(test[column])
            continue
        values = list(set(train[column].tolist()))
        df = OHE_Function(column, values, test)
        test_ohe=test_ohe.join(df)
    del test_ohe['Index']
    
    return train_ohe, actions_ohe, test_ohe

def train_model(true, false):                                               #Model training
    
    models = {}
    
    start = time.time()
    for i in range(50):
        
        t = true.sample(n=500)
        f = false.sample(n=500)
        data = pd.concat([t, f])
        Y = data['REWARD']
        print(sum(Y), len(Y)-sum(Y))
        X = data.drop(['REWARD'], axis=1)
        lr = LogisticRegression(max_iter=10000).fit(X, Y)
        models[i] = lr
    end = time.time()
    print(end-start)
    return models

train = pd.read_csv('Cohort.csv')
test = pd.read_excel('test.xlsx')
actions = pd.read_excel('Actions.xlsx')

train_ohe, actions_ohe, test_ohe = Data_PreProcessing(train, actions, test)
TRUE = train_ohe[train_ohe['REWARD'] == 1]
FALSE = train_ohe[train_ohe['REWARD'] == 0]


#Prediction code starts from here.
models = train_model(TRUE, FALSE)
result = []                             #Saves prediction for user in the test data
for user in range(len(test_ohe)):
    
    user_result = []
    
    for arm in range(len(actions_ohe)):
        
        model_result = []
        inp = test_ohe.iloc[user].tolist()+actions_ohe.iloc[arm].tolist()
        
        for model in models:
             
            out = models[model].predict_proba([inp])[0][1]
            model_result.append(out)
        
        model_result = sorted(model_result, reverse = True)
        max_of_95_percentile = model_result[5]
        user_result.append((arm, max_of_95_percentile))

    user_result = sorted(user_result, reverse =True, key = lambda x: x[1])
    result.append(user_result[0])
