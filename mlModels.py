import pandas as pd
import numpy as np
import pickle
import  xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


df= pd.read_csv("data/quora_features.csv",sep=',')

#Create a new dataFrame
X= pd.DataFrame()
y=df['is_duplicate']

#segregating different features
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
f6=feature_columns_1+feature_columns_3+feature_columns_4

def train_model(features,X,y):
    for column in features:
        X[column] = df[column]
        X[column].replace(np.inf, np.nan, inplace=True)
        X[column].fillna(X[column].mean(), inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logistic_model = LogisticRegression().fit(X,y)
    xgb_model = xgb.XGBClassifier().fit(X, y)

    with open('data/logistic_model_'+features, 'wb') as fid:
        pickle.dump(logistic_model, fid, 2)
    with open('data/xgb_model_'+features, 'wb') as fid:
        pickle.dump(xgb_model, fid, 2)

#calling function to train according to different features
for i in range(1,7):
    train_model('f'+i,X,y)
















