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
user = pd.read_excel('Hist.xlsx')
test = pd.read_excel('test.xlsx')

actions = actions.drop(['plan_id', 'spark_plan_name', 'unique_tag'], axis=1)
user = user.drop(['user_id', 'enrolled spark plan'], axis=1)
test = test.drop(['user_id'], axis=1)
actions = actions.fillna('missing')
user = user.fillna('missing')
test = test.fillna('missing')

def ohe_actions(actions):
    
    action_set = pd.DataFrame( np.zeros((26, 10)) ,columns = ['AMOUNT', 'LMF', 'DIGITAL_GOLD', 'INSURANCE', 'TAX', 'CREDIT', 'MEDIUM PLAN', 'MINI PLAN', 'LARGE PLAN', 'INVESTMENT'])
    for row in range(26):
    
        for col in actions.columns[1:]:
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
    
    
action = ohe_actions(actions)
hist = ohe_user_data(user)
hist = hist.dropna()
user = ohe_test(test)
user['INCOME'] = np.random.randint(20, 100, size=(100, 1))
user['DEPENDENTS'] = np.random.randint(0, 5, size=(100, 1))
action = action.drop(['MEDIUM PLAN', 'MINI PLAN', 'LARGE PLAN', 'INVESTMENT'], axis=1)
hist = hist.drop(['MEDIUM PLAN', 'MINI PLAN', 'LARGE PLAN', 'INVESTMENT'], axis=1)



def create_random_zeros(X):
    df1 = pd.DataFrame(np.random.randint(0, 2, size=(180, 1)), columns=['MALE'])
    df2 = pd.DataFrame([1-i for i in df1['MALE'].tolist()], columns=['FEMALE'])
    df3 = pd.DataFrame(np.random.randint(20, 50, size=(180, 1)), columns=['AGE'])
    df4 = pd.DataFrame(np.random.randint(20, 100, size=(180, 1)), columns=['INCOME'])
    df5 = pd.DataFrame(np.random.randint(0,5, size = (180, 1)), columns=['DEPENDENTS'])
    df6 = pd.DataFrame(np.random.randint(0, 2, size = (180, 1)), columns=['AMOUNT'])
    df7 = pd.DataFrame(np.random.randint(0, 2, size = (180, 1)), columns=['LMF'])
    df8 = pd.DataFrame(np.random.randint(0, 2, size = (180, 1)), columns=['DIGITAL_GOLD'])
    df9 = pd.DataFrame(np.random.randint(0, 2, size = (180, 1)), columns=['INSURANCE'])
    df10 = pd.DataFrame(np.random.randint(0, 2, size = (180, 1)), columns=['TAX'])
    df11 = pd.DataFrame(np.random.randint(0, 2, size = (180, 1)), columns=['CREDIT'])
    df16 = pd.DataFrame(np.zeros((180, 1)), columns = ['REWARD'])
    Y = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df16], axis=1)
    X = X.append(Y, ignore_index=True)
    return X
    
    
def initialize_models(Z):  
    
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr1 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr2 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr3 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr4 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr5 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr6 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr7 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr8 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr9 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    lr10 = LogisticRegression(max_iter=1000).fit(X, Y)
    X = create_random_zeros(Z)
    X = X.sample(frac=1).reset_index(drop=True)
    Y = X['REWARD']
    X = X.drop(['REWARD'], axis=1)
    
    return [lr1, lr2, lr3, lr4, lr5, lr6, lr7, lr8, lr9, lr10]
    
    
    
result = []
for user_feature in range(len(user)):
    user_result = []
    models = initialize_models(hist)
    
    for action_feature in range(len(action)):
        inp = pd.concat([user.loc[user_feature], action.loc[action_feature]])
        
        for model in models:
            out = model.predict([inp])
            if out[0] == 1:
                user_result.append(action_feature)
    result.append(user_result)


arm_to_recommend = []
for new in range(100):

    if len(result[new]) != 0:
        
        freq = collections.Counter(result[new])
        for key in freq:
            if freq[key] < 10:
                arm_to_recommend.append((new, key)) 
                break
