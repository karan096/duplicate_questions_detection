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

def predict(feature_columns,X_test):
    for column in feature_columns:
        X_test[column] = test[column]

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    file = open('data/logistic_model_f1+f3.pkl', 'rb')
    logistic_model = pickle.load(file)
    print(logistic_model.predict_proba(X_test))



predict(feature_columns_1+feature_columns_3,X_test)
















