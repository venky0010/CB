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

customer_features = ['AGE', 'GENDER', 'INCOME/PM',  'DEPENDENTS', 'COMPANY']
product_features = ['COMPONENTS', 'AMOUNT', 'PLAN_SIZE', 'MEDIUM']
features = {'AGE' : 0,
            'GENDER' : ['MALE', 'FEMALE', 'OTHERS', 'missing'],
            'INCOME/PM': 0,
            'DEPENDENTS': 0,
            'COMPANY': ['RENTEL CAR', 'KONASTSH (VISTARA VENDOR)', 'DMART', 'ZOMATO', 'MEGHA CAB', 'INK', 'MEGA CABS', 'DELHI AIRPORT', 'AIRPORT TAXI KSTDC', 'VANDER KTC', 
                        'CORPORATE CABS', 'VODAFONE', 'RADIO CABS', 'AIRPORT TAXI', 'KONASTH', 'SREE', 'PAID', 'VAISNAVI INTIRIAL', 'BSNL', 'MISSING', 'BEL FILING', 
                        'LINKED', 'SKY CABS', 'BIAL TERMINAL', 'SREE EMPOLYEE', 'SHANTHINIKETHANA', 'PREPAID', 'VISTARA AIRLINES', 'POWER GAS', 
                        'DELHI AIRPORT SERVICE PVT', 'SCOPE', 'CARZONE RENT', 'MEGHA', 'KAALI PEELI', 'DP', 'BAJAJ', 'ECOSPACE', 'NOT REACHABLE', 'VINAY', 'VISTARA', 
                        'DELHI AIRPORTS SERVICE', 'KONASTSH', 'CAR ZONE', 'SKY CAB', 'OWN VEHICLE', 'KSTDC', 'CAMPAIGN', 'SS2 INTERVALS', 'TRAVEL AGENCY', 
                        'KTC INDIA PVT LTD', 'ACENGER', 'OLA/UBER', 'EGL', 'SWITCHED OFF', 'KIRLOKSKAR', 'RED', 'COGNIZANT', 
                        'INDEPENDENT', 'SALARY', 'OTHERS', 'LTT', 'FIRST CARS', 'GHMC', 'FIDELITY MANYATA TECH PARK NAGAVARA', 
                        'SHELL CABS', 'AIR INDIA', 'TAXI', 'DELHI AIRPORT SERVICE', 'VISTARA AIR LINSE', 'SKY', 'FANCY STORE & CAB'],
            'COMPONENTS': ['CREDIT_', 'GOLD_', 'INSURANCE_', 'LMF_', 'TAX_', 'INSURANCE_LMF_GOLD_', 'LMF_GOLD_', 'TAX_CREDIT_',
                             'TAX_GOLD_', 'TAX_INSURANCE_LMF_GOLD_', 'TAX_LMF_', 'TAX_LMF_GOLD_', 'TAX_INSURANCE_'],
            'AMOUNT': ['500', '1000', '2000', '3000', '4000', '5000'],
            'PLAN_SIZE': ['MINI', 'MEDIUM', 'LARGE', 'EXTRA LARGE'],
            'MEDIUM': ['CALL', 'SMS', 'WHATSAPP', 'IVR', 'PUSH NOTIFICATIONS']
            }

def OHE_Function(column, values, data):
    
    c = []
    for val in values:
        val = str(val).upper()+str('_')+str(column)
        c.append(val)
    
    dummy = pd.DataFrame(np.zeros((len(data), len(values))), columns=c)
    
    for i in range(len(data)):
        val = data.loc[i, column]
        col = str(val).upper()+str('_')+str(column)
        dummy.loc[i, col] = 1
        
    return dummy
  
def OHE(train, test, actions):
    
    #OHE TRAINING DATA
    train_ohe = pd.DataFrame([i for i in range(len(train))], columns=['Index'])
    for column in features:
        values = features[column]
        if values == 0:
            train_ohe=train_ohe.join(train[column])
            continue
        df = OHE_Function(column, values, train)
        train_ohe=train_ohe.join(df)
        
    train_ohe = feature_interaction(train_ohe)
    train_ohe = train_ohe.join(train['REWARD'])
    del train_ohe['Index']
    
    #OHE TEST DATA
    test_ohe = pd.DataFrame([i for i in range(len(test))], columns=['Index'])
    for column in ['AGE', 'GENDER', 'INCOME/PM', 'DEPENDENTS', 'COMPANY']:
        values = features[column]
        if values == 0:
            test_ohe=test_ohe.join(test[column])
            continue
        df = OHE_Function(column, values, test)
        test_ohe=test_ohe.join(df)
    del test_ohe['Index']
    
    #OHE Actions data
    actions_ohe = pd.DataFrame([i for i in range(len(actions))], columns=['Index'])
    for column in ['COMPONENTS', 'AMOUNT', 'PLAN_SIZE', 'MEDIUM']:
        values = features[column]
        print(values)
        if values == 0:
            actions_ohe=actions_ohe.join(actions[column])
            continue
        df = OHE_Function(column, values, actions)
        actions_ohe=actions_ohe.join(df)
    del actions_ohe['Index']  
    
    return train_ohe, test_ohe, actions_ohe
  
#Create Random Reward = 0 Data
def Create_Reward_Zero_data(train, features, l):
    
    df = pd.DataFrame([i for i in range(l)], columns=['Index'])
    for feature in features:
        values = features[feature]
        if feature == 'AGE':
            df[feature] = np.random.randint(20, 60, l)
            continue
        if feature == 'INCOME/PM':
            df[feature] = np.random.randint(10, 100, l)
            continue
        if feature == 'DEPENDENTS':
            df[feature] = np.random.randint(0, 3, l)
            continue 
            
        numbers = np.random.randint(0, len(values), l)
        for i in range(len(numbers)):
            df.loc[i, feature] = values[numbers[i]]
    
    df['REWARD'] = [0]*l
    del df['Index']
    
    return df
  
def feature_interaction(data):
    
    user_columns = ['AGE', 'MALE_GENDER', 'FEMALE_GENDER', 'OTHERS_GENDER', 'MISSING_GENDER', 'INCOME/PM', 'DEPENDENTS', 
                    'RENTEL CAR_COMPANY', 'KONASTSH (VISTARA VENDOR)_COMPANY', 'DMART_COMPANY', 'ZOMATO_COMPANY', 
                    'MEGHA CAB_COMPANY', 'INK_COMPANY', 'MEGA CABS_COMPANY', 'DELHI AIRPORT_COMPANY', 'AIRPORT TAXI KSTDC_COMPANY', 
                    'VANDER KTC_COMPANY', 'CORPORATE CABS_COMPANY', 'VODAFONE_COMPANY', 'RADIO CABS_COMPANY', 'AIRPORT TAXI_COMPANY', 
                    'KONASTH_COMPANY', 'SREE_COMPANY', 'PAID_COMPANY', 'VAISNAVI INTIRIAL_COMPANY', 'BSNL_COMPANY', 'MISSING_COMPANY', 
                    'BEL FILING_COMPANY', 'LINKED_COMPANY', 'SKY CABS_COMPANY', 'BIAL TERMINAL_COMPANY', 'SREE EMPOLYEE_COMPANY', 
                    'SHANTHINIKETHANA_COMPANY', 'PREPAID_COMPANY', 'VISTARA AIRLINES_COMPANY', 'POWER GAS_COMPANY', 'DELHI AIRPORT SERVICE PVT_COMPANY', 
                    'SCOPE_COMPANY', 'CARZONE RENT_COMPANY', 'MEGHA_COMPANY', 'KAALI PEELI_COMPANY', 'DP_COMPANY', 'BAJAJ_COMPANY', 
                    'ECOSPACE_COMPANY', 'NOT REACHABLE_COMPANY', 'VINAY_COMPANY', 'VISTARA_COMPANY', 'DELHI AIRPORTS SERVICE_COMPANY', 
                    'KONASTSH_COMPANY', 'CAR ZONE_COMPANY', 'SKY CAB_COMPANY', 'OWN VEHICLE_COMPANY', 'KSTDC_COMPANY', 'CAMPAIGN_COMPANY', 
                    'SS2 INTERVALS_COMPANY', 'TRAVEL AGENCY_COMPANY', 'KTC INDIA PVT LTD_COMPANY', 'ACENGER_COMPANY', 'OLA/UBER_COMPANY', 
                    'EGL_COMPANY', 'SWITCHED OFF_COMPANY', 'KIRLOKSKAR_COMPANY', 'RED_COMPANY', 'COGNIZANT_COMPANY', 'INDEPENDENT_COMPANY', 
                    'SALARY_COMPANY', 'OTHERS_COMPANY', 'LTT_COMPANY', 'FIRST CARS_COMPANY', 'GHMC_COMPANY', 'FIDELITY MANYATA TECH PARK NAGAVARA_COMPANY', 
                    'SHELL CABS_COMPANY', 'AIR INDIA_COMPANY', 'TAXI_COMPANY', 'DELHI AIRPORT SERVICE_COMPANY', 'VISTARA AIR LINSE_COMPANY', 
                    'SKY_COMPANY', 'FANCY STORE & CAB_COMPANY', 'MISSING_COMPANY']
    product_columns = ['CREDIT__COMPONENTS', 'GOLD__COMPONENTS', 'INSURANCE__COMPONENTS', 'LMF__COMPONENTS', 'TAX__COMPONENTS', 
                       'INSURANCE_LMF_GOLD__COMPONENTS', 'LMF_GOLD__COMPONENTS', 'TAX_CREDIT__COMPONENTS', 'TAX_GOLD__COMPONENTS', 
                       'TAX_INSURANCE_LMF_GOLD__COMPONENTS', 'TAX_LMF__COMPONENTS', 'TAX_LMF_GOLD__COMPONENTS', 'TAX_INSURANCE__COMPONENTS', 
                       '500_AMOUNT', '1000_AMOUNT', '2000_AMOUNT', '3000_AMOUNT', '4000_AMOUNT', '5000_AMOUNT', 'MINI_PLAN_SIZE',
                       'MEDIUM_PLAN_SIZE', 'LARGE_PLAN_SIZE', 'EXTRA LARGE_PLAN_SIZE', 'CALL_MEDIUM', 'SMS_MEDIUM', 'WHATSAPP_MEDIUM', 
                       'IVR_MEDIUM', 'PUSH NOTIFICATIONS_MEDIUM']
    for col1 in user_columns:
        for col2 in product_columns:
            print(col1, col2)
            data[str(col1)+'_'+str(col2)] = data[col1]*data[col2]
    return data
  
def train_model(data):
    
    models = {}
    start = time.time()
    for i in range(50):
        
        Y = data['REWARD']
        print(sum(Y), len(Y)-sum(Y))
        X = data.drop(['REWARD'], axis=1)
        lr = LogisticRegression(max_iter=10000).fit(X, Y)
        models[i] = lr
        
    end = time.time()
    print(end-start)
    return models


  
  
def RUN(train, test, actions):
    
    train_zero = Create_Reward_Zero_data(train, features, 500)
    train_one = train.sample(n=500)
    train_final = pd.concat([train_one, train_zero], ignore_index=True)
    train_ohe, test_ohe, actions_ohe = OHE(train_final, test, actions)
    models = train_model(train_ohe)
            
    result = []
    for user in range(len(test_ohe)):
    
        user_result = []
        for arm in range(len(actions_ohe)):
        
            model_result = []
            test = test_ohe.iloc[user].tolist()
            action = actions_ohe.iloc[arm].tolist()
            inp = test + action
            interactions = []
            for i in test:
                for j in action:
                    inp.append(i*j)
            
            for model in models:
             
                out = models[model].predict_proba([inp])[0][1]
                model_result.append(out)
        
            model_result = sorted(model_result, reverse = True)
            max_of_95_percentile = model_result[5]
            user_result.append((arm, max_of_95_percentile))

        user_result = sorted(user_result, reverse =True, key = lambda x: x[1])
        result.append(actions.loc[user_result[0][0]])
    return result
  
result = RUN(train_ohe, test_ohe, actions_ohe)
