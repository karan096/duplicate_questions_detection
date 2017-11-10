import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

test=pd.read_csv('data/quora_features_exp.csv',sep=',')
X_test=pd.DataFrame()

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
f6=feature_columns_1+feature_columns_3+feature_columns_4

def predict(features,X_test):
    for column in features:
        X_test[column] = test[column]

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    logistic_file = open('data/logistic_model_'+features, 'rb')
    xgb_file= open('data/xgb_model_'+features, 'rb')

    logistic_model = pickle.load(logistic_file)
    xgb_model=pickle.load(xgb_file)

    print(logistic_model.predict_proba(X_test))
    print(xgb_model.predict_proba(X_test))

for i in range(1,7):
    predict('f'+i,X_test)