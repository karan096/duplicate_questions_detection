import pandas as pd
import  numpy as np
import pickle
from flask import jsonify
from sklearn.preprocessing import StandardScaler


#assign required columns to the data frame
feature_columns_1=['len_q1','len_q2','diff_len','len_char_q1','len_char_q2','len_word_q1','len_word_q2','common_words']
feature_columns_2=['fuzz_qratio','fuzz_WRatio','fuzz_partial_ratio','fuzz_partial_token_set_ratio','fuzz_partial_token_sort_ratio',
                   'fuzz_token_set_ratio','fuzz_token_sort_ratio']
feature_columns_3=['wmd','norm_wmd','cosine_distance','cityblock_distance','jaccard_distance','canberra_distance',
                 'euclidean_distance','minkowski_distance','braycurtis_distance','skew_q1vec','skew_q2vec','kur_q1vec','kur_q2vec']
feature_columns_4=['cosSim']

f1=feature_columns_1
f2=feature_columns_1+feature_columns_2
f3=feature_columns_1+feature_columns_2+feature_columns_3
f4=feature_columns_1+feature_columns_4
f5=feature_columns_1+feature_columns_2+feature_columns_4
# f6=feature_columns_4

f=[f1,f2,f3,f4,f5]

predictionJSON={}
# features=['Basic','Basic+Fuzzy','Basic + Fuzzy + word distances','Basic+ word distances',
#           'Basic+ Fuzzy+ Cosine Similarity','Basic+word distances+ Cosine Similarity']
predictionJSON['Logistic Regression']=[]
predictionJSON['XG_Boost']=[]

def predict(features,X_test,features_name,test):
    #print(test)
    #print(X_test)
    for column in features:
        X_test[column]=test[column]
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    logistic_file = open('data/logistic_model_'+str(features_name), 'rb')
    xgb_file= open('data/xgb_model_'+str(features_name), 'rb')

    logistic_model = pickle.load(logistic_file)
    xgb_model=pickle.load(xgb_file)

    predictionJSON['Logistic Regression'].append(str(list(logistic_model.predict_proba(X_test))[0][1]))
    predictionJSON['XG_Boost'].append(str(list(xgb_model.predict_proba(X_test))[0][1]))

def predictProbabilityForDifferentFeatures():

    #print (X_test)
    test = pd.read_csv('data/quora_features_test.csv', sep=',')
    # h1=test.columns.values.tolist()
    # # for column in h1:
    # #      test[column].replace(np.inf,np.nan,inplace=True)
    # #      test[column].fillna(test[column].mean(), inplace=True)
    # # with open('data/quora_features_test.csv') as f1:
    # #      last_row = f1.readlines()[-1]
    # last_row=test.tail(n=1)

    # print(pd.read_csv('data/quora_features_test.csv', nrows=1))
    # test = data.drop([''], axis=1)

    # X_test=test.columns.values.tolist()
    # for feature,value in zip(h1,last_row):
    #     X_test[feature]=value
    # #print(X_test)
    # print(last_row)
    for column in f1+f2+f3+f4+f5:
        test[column].replace(np.inf, np.nan, inplace=True)
        test[column].fillna(test[column].mean(), inplace=True)

    test=test.drop(['id','qid1','qid2','question1','question2'],axis=1)
    test=test.tail(n=1)
    # X_test = test.tail(n=1)
    # X_test=X_test.drop(['id','qid1','qid2','question1','question2'],axis=1)
    # print(X_test)
    X_test=pd.DataFrame()
    for i in range(0, 5):
        predict(f[i], X_test, str('f') + str(i + 1),test)

    #print (predictionJSON)
    return predictionJSON

predictProbabilityForDifferentFeatures()