import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from sklearn.linear_model import LogisticRegression

%matplotlib inline


actions = pd.read_excel('Actions.xlsx')
user = pd.read_excel('user_data.xlsx')
test = pd.read_excel('test.xlsx')

actions = actions.fillna('missing')
user = user.fillna('missing')
test = test.fillna('missing')
actions = actions.drop(['plan_id', 'spark_plan_name', 'unique_tag'], axis=1)
user = user.drop(['user_id', 'unique_tag'], axis=1)
test = test.drop(['user_id'], axis=1)

def ohe_actions(actions):
    
    action_set = pd.DataFrame(np.zeros((26, 10)) ,columns = ['AMOUNT', 'LMF', 'DIGITAL_GOLD', 'INSURANCE', 'TAX', 'CREDIT', 'MEDIUM PLAN', 'MINI PLAN', 'LARGE PLAN', 'INVESTMENT'])
    for row in range(26):
    
        for col in actions.columns:
                                                                #Amount
            if col == 'amount':                     
                val = actions.loc[row, col]
                if val == 'missing':
                    continue
                action_set.loc[row, 'AMOUNT'] = val
                                                                #One Hot Encode tag
            elif col == 'tag':           
                val = actions.loc[row, col]
                if val == 'missing':
                    continue
                action_set.loc[row, 'INVESTMENT'] = 1
                                                                #OHE Components
            elif col == 'components':            
                val = actions.loc[row, col]           
                if re.search("lmf", val):
                    action_set.loc[row, 'LMF'] = 1            
                if re.search("digital_gold", val):
                    action_set.loc[row, 'DIGITAL_GOLD'] = 1                   
                if re.search("insurance", val):
                    action_set.loc[row, 'INSURANCE'] = 1                 
                if re.search("tax", val):
                    action_set.loc[row, 'TAX'] = 1             
                if re.search("credit", val):
                    action_set.loc[row, 'CREDIT'] = 1
                                                                #OHE Plan size
            elif col == 'plan_size':
                
                val = actions.loc[row, col]
                if val == 'missing':
                    continue
                elif val == 'Medium Plan':
                    action_set.loc[row, 'MEDIUM PLAN'] = 1
                elif val == 'Mini Plan':
                    action_set.loc[row, 'MINI PLAN'] = 1
                elif val == 'Large Plan':
                    action_set.loc[row, 'LARGE PLAN'] = 1
                    
    return action_set


def ohe_user_data(user):        #Historical data
    
    X = pd.DataFrame(np.zeros((20, 16)), columns = ['MALE', 'FEMALE', 'AGE', 'INCOME', 'DEPENDENTS', 'AMOUNT', 'LMF', 'DIGITAL_GOLD', 'INSURANCE', 'TAX', 'CREDIT', 'MEDIUM PLAN', 'MINI PLAN', 'LARGE PLAN', 'INVESTMENT', 'REWARD'])
    for row in range(21):
    
        X.loc[row, 'REWARD'] = 1
    
        for col in user.columns:
        
            if col == 'gender':
                val = user.loc[row, col]
                if val == 'missing':
                    continue
                elif val == 'male':
                    X.loc[row, 'MALE'] = 1
                elif val == 'female':
                    X.loc[row, 'FEMALE'] = 1

                
            elif col == 'age':
                val = user.loc[row, col]
                if val == 'missing':
                    continue
                X.loc[row, 'AGE'] = val
            
            elif col == 'income/pm':
                val = user.loc[row, col]
                if val == 'missing':
                    continue
                X.loc[row, 'INCOME'] = val
            
            elif col == 'dependents':
                val = user.loc[row, col]
                if val == 'missing':
                    continue
                X.loc[row, 'DEPENDENTS'] = val
            
            elif col == 'gold':
                val = user.loc[row, col]
                if val=='missing':
                    continue
                X.loc[row, 'DIGITAL_GOLD']=1
            
            elif col == 'lmf':
                val = user.loc[row, col]
                if val=='missing':
                    continue
                X.loc[row, 'LMF']=1
            
            elif col == 'insurance':
                val = user.loc[row, col]
                if val=='missing':
                    continue
                X.loc[row, 'INSURANCE']=1
            
            elif col == 'credit':
                val = user.loc[row, col]
                if val=='missing':
                    continue
                X.loc[row, 'CREDIT']=1
            
            elif col == 'tax':
                val = user.loc[row, col]
                if val=='missing':
                    continue
                X.loc[row, 'TAX']=1
            
            elif col == 'amount':
                val = user.loc[row, col]
                X.loc[row, 'AMOUNT'] = val
                
            elif col == 'plan_size':
                val = user.loc[row, col]
                if val == 'Mini Plan':
                    X.loc[row, 'MINI PLAN'] = 1
                elif val == 'Medium Plan':
                    X.loc[row, 'MEDIUM PLAN'] = 1
                elif val == 'Large Plan':
                    X.loc[row, 'LARGE PLAN'] = 1
            
            elif col == 'tag':
                val = user.loc[row, col]
                X.loc[row, 'INVESTMENT'] = 1
    return X


def ohe_test(test):
    
    user_features = pd.DataFrame(np.zeros((100, 5)), columns = ['MALE', 'FEMALE', 'AGE', 'INCOME', 'DEPENDENTS'])  

    for row in range(len(test)):
        for col in test.columns:   
            if col == 'gender':
                val = test.loc[row, col]
                if val=='missing':
                    continue
                if val == 'male':
                    user_features.loc[row, 'MALE'] =1              
                if val == 'female':
                    user_features.loc[row, 'FEMALE'] =1
            elif col == 'age':
                val = test.loc[row, col]
                user_features.loc[row, 'AGE'] = val
            elif col == 'income/pm':
                val = test.loc[row, col]
                if val == 'missing':
                    continue
                user_features.loc[row, 'INCOME'] = val
            elif col == 'dependents':
                val = test.loc[row, col]
                if val == 'missing':
                    continue
                user_features.loc[row, 'DEPENDENTS'] = val
                
    return user_features


for i in range(1, 20):
    
    for j in range(-2, 3):
    
        x = hist.loc[i]
        x[2]+=j
        hist = hist.append(x, ignore_index=True)
        
    for j in range(-1, 2):
        x = hist.loc[i]
        x[3]+=j
        hist = hist.append(x, ignore_index=True)
        
        
def initialize_models(Z, user, action):  
    
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr1 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr2 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr3 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr4 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr5 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr6 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr7 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr8 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr9 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr10 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr11 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr12 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr13 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr14 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr15 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr16 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr17 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr18 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr19 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr20 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr21 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr22 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr23 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr24 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr25 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr26 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr27 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr28 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr29 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr30 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr31 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr32 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr33 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr34 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr35 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr36 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr37 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr38 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr39 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr40 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr41 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr42 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr43 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr44 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr45 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr46 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr47 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr48 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr49 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr50 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr51 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr52 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr53 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr54 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr55 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr56 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr57 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr58 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr59 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr60 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr61 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr62 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr63 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr64 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr65 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr66 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr67 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr68 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr69 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr70 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr71 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr72 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr73 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr74 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr75 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr76 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr77 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr78 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr79 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr80 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr81 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr82 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr83 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr84 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr85 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr86 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr87 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr88 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr89 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr90 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr91 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr92 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr93 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr94 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr95 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr96 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr97 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr98 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr99 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z, user, action)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr100 = LogisticRegression(max_iter=1000).fit(X, Y)
   
    
    return [lr1, lr2, lr3, lr4, lr5, lr6, lr7, lr8, lr9, lr10,
            lr11, lr12, lr13, lr14, lr15, lr16, lr17, lr18, lr19, lr20,
            lr21, lr22, lr23, lr24, lr25, lr26, lr27, lr28, lr29, lr30,
            lr31, lr32, lr33, lr34, lr35, lr36, lr37, lr38, lr39, lr40,
            lr41, lr42, lr43, lr44, lr45, lr46, lr47, lr48, lr49, lr50,
            lr51, lr52, lr53, lr54, lr55, lr56, lr57, lr58, lr59, lr60,
            lr61, lr62, lr63, lr64, lr65, lr66, lr67, lr68, lr69, lr60,
            lr71, lr72, lr73, lr74, lr75, lr76, lr77, lr78, lr79, lr80,
            lr81, lr82, lr83, lr84, lr85, lr86, lr87, lr88, lr89, lr90,
            lr91, lr92, lr93, lr94, lr95, lr96, lr97, lr98, lr99, lr100]

def create_random_zeros(X, user, action):

    df1 = pd.DataFrame(np.random.randint(0, 2, size=(1000, 1)), columns=['MALE'])
    df2 = pd.DataFrame([1-i for i in df1['MALE'].tolist()], columns=['FEMALE'])
    df3 = pd.DataFrame(np.random.randint(20, 50, size=(1000, 1)), columns=['AGE'])
    df4 = pd.DataFrame(np.random.randint(20, 100, size=(1000, 1)), columns=['INCOME'])
    df5 = pd.DataFrame(np.random.randint(0,5, size = (1000, 1)), columns=['DEPENDENTS'])
    df6 = pd.DataFrame(np.random.randint(0, 2, size = (1000, 1)), columns=['AMOUNT'])
    df7 = pd.DataFrame(np.random.randint(0, 2, size = (1000, 1)), columns=['LMF'])
    df8 = pd.DataFrame(np.random.randint(0, 2, size = (1000, 1)), columns=['DIGITAL_GOLD'])
    df9 = pd.DataFrame(np.random.randint(0, 2, size = (1000, 1)), columns=['INSURANCE'])
    df10 = pd.DataFrame(np.random.randint(0, 2, size = (1000, 1)), columns=['TAX'])
    df11 = pd.DataFrame(np.random.randint(0, 2, size = (1000, 1)), columns=['CREDIT'])
    
    medium = [i[0] for i in np.random.randint(0, 2, size=(1000, 1))]
    x = [i[0] for i in np.random.randint(0, 2, size=(1000, 1))]
    mini = [x[i] if medium[i] == 0 else 0 for i in range(len(x))]
    large = [1 if medium[i]+mini[i]==0 else 0 for i in range(len(x))]
    
    df12 = pd.DataFrame(medium, columns=['MEDIUM PLAN'])
    df13 = pd.DataFrame(mini, columns=['MINI PLAN'])
    df14 = pd.DataFrame(large, columns=['LARGE PLAN'])
    df15 = pd.DataFrame(np.random.randint(1, 2, size = (1000, 1)), columns=['INVESTMENT'])
    df16 = pd.DataFrame(np.zeros((1000, 1)), columns = ['REWARD'])
    Y = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16], axis=1)
    X = X.append(Y, ignore_index=True)
    
    for col1 in user.columns:
        for col2 in action.columns:
            col_name = col1+"_"+col2
            x = X.loc[:, col1]*X.loc[:, col2]
            y=pd.DataFrame([i for i in x], columns=[col_name])
            X = pd.concat([X, y], axis=1)
    
    return X


result = []
models = initialize_models(hist, user, action)

for user_feature in range(len(user)):
    
    user_result = []
    
    for arm_feature in range(len(action)):
        
        model_result = []
        inp = []
        for i in user.loc[user_feature]:
            inp.append(i)
        for i in action.loc[arm_feature]:
            inp.append(i)
        for i in user.loc[user_feature]:
            for j in action.loc[arm_feature]:
                inp.append(i*j)
        #inp = pd.concat([user.loc[user_feature], action.loc[arm_feature]])

        for model in models:
            
            out = model.predict_proba([inp])[0][1]
            model_result.append(out)
            
        model_result = sorted(model_result, reverse=True)
        max_of_95th_percentile = model_result[6]
        user_result.append((arm_feature, max_of_95th_percentile))
    
    user_result = sorted(user_result,reverse = True, key = lambda x: x[1])
    print(user_result)
    result.append(user_result[:5])
    
    
    
import random

actions_ = pd.read_excel('Actions.xlsx')
reco = []
for i in result:
    index = []
    for j in i:
        index.append(j[0])
    recom = random.choice(index)
    reco.append(recom)
    
user['Recommendation'] = [0]*100
for i in range(100):
    user.loc[i, 'Recommendation']=reco[i]
    
for i in range(len(user)):
    tag = actions_.loc[user.loc[i, 'Recommendation'], 'unique_tag']
    user.loc[i, 'Recommended_Unique_tag']=tag
    
    
user.to_csv('Reco.csv')
