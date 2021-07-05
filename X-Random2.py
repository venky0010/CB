import re
import math
import random
from random import choice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from sklearn.linear_model import LogisticRegression

%matplotlib inline

train = pd.read_csv('Cohort.csv')
test = pd.read_excel('test.xlsx')
actions = pd.read_excel('Actions.xlsx')

actions = actions.fillna(0)
test = test.fillna('missing')
actions = actions.drop(['plan_id', 'spark_plan_name', 'unique_tag', 'tag', 'components'], axis=1)
test = test.drop(['user_id'], axis=1)

for i in range(len(train)//2, len(train)):
    train.loc[i, 'REWARD'] = 0
    
def OHE(column, values, train):
    
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
  
  
  train_ohe = pd.DataFrame([i for i in range(len(train))], columns=['Index'])
#columns = ['GENDER', 'COMPANY', 'PLAN_SIZE', 'MEDIUM']
for column in train.columns:
    if type(train[column][0]) == np.int64:
        train_ohe=train_ohe.join(train[column])
        continue
    values = list(set(train[column].tolist()))
    df = OHE(column, values, train)
    train_ohe=train_ohe.join(df)
del train_ohe['Index']

actions_ohe = pd.DataFrame([i for i in range(len(actions))], columns=['Index'])
for column in actions.columns:
    if type(actions[column][0]) == np.int64 or column in ['CREDIT', 'TAX','LMF', 'INSURANCE', 'GOLD']:
        actions_ohe=actions_ohe.join(actions[column])
        continue
    values = list(set(actions[column].tolist()))
    df = OHE(column, values, actions)
    actions_ohe=actions_ohe.join(df)
del actions_ohe['Index']

test_ohe = pd.DataFrame([i for i in range(len(test))], columns=['Index'])
for column in test.columns:
    if type(test[column][0]) == np.int64:
        test_ohe=test_ohe.join(test[column])
        continue
    values = list(set(train[column].tolist()))
    df = OHE(column, values, test)
    test_ohe=test_ohe.join(df)
del test_ohe['Index']


TRUE = train_ohe[train_ohe['REWARD'] == 1]
FALSE = train_ohe[train_ohe['REWARD'] == 0]
print(len(TRUE), len(FALSE))

import time
def train_model(true, false):
    
    models = {}
    t = true.sample(n=1000)
    f = false.sample(n=1000)
    data = pd.concat([t, f])

    Y = data['REWARD']
    print(sum(Y), len(Y)-sum(Y))
    X = data.drop(['REWARD'], axis=1)
    
    start = time.time()
    for i in range(100):
        lr = LogisticRegression(max_iter=10000).fit(X, Y)
        models[i] = lr
    end = time.time()
    print(end-start)
    return models
  
result = []



for user in range(len(test_ohe)):
    
    user_result = []
    
    for arm in range(len(actions_ohe)):
        
        model_result = []
        inp = test_ohe.iloc[user].tolist()+actions_ohe.iloc[arm].tolist()
        print(inp)
        for model in models:
            print(model)
            out = models[model].predict_proba([inp])[0][1]
            model_result.append(out)
