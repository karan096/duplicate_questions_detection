import pandas as pd
import numpy as np
import pickle
import  xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


df= pd.read_csv("data/quora_features.csv",sep=',')
test=pd.read_csv("data/quora_features_exp.csv",sep=',')


#Create a new dataFrame
X= pd.DataFrame()
y=df['is_duplicate']
X_test=pd.DataFrame()

#assign required columns to the data frame
feature_columns_1=['len_q1','len_q2','diff_len','len_char_q1','len_char_q2','len_word_q1','len_word_q2','common_words']
feature_columns_2=['fuzz_qratio','fuzz_WRatio','fuzz_partial_ratio','fuzz_partial_token_set_ratio','fuzz_partial_token_sort_ratio',
                   'fuzz_token_set_ratio','fuzz_token_sort_ratio']
feature_columns_3=['wmd','norm_wmd','cosine_distance','cityblock_distance','jaccard_distance','canberra_distance',
                 'euclidean_distance','minkowski_distance','braycurtis_distance','skew_q1vec','skew_q2vec','kur_q1vec','kur_q2vec']
# feature_columns=['len_q1','len_q2','diff_len','len_char_q1','len_char_q2','len_word_q1','len_word_q2','common_words','fuzz_qratio',
#                 'fuzz_WRatio','fuzz_partial_ratio','fuzz_partial_token_set_ratio','fuzz_partial_token_sort_ratio','fuzz_token_set_ratio',
#                  'fuzz_token_sort_ratio','wmd','norm_wmd','cosine_distance','cityblock_distance','jaccard_distance','canberra_distance',
#                  'euclidean_distance','minkowski_distance','braycurtis_distance','skew_q1vec','skew_q2vec','kur_q1vec','kur_q2vec']

for column in feature_columns_3+feature_columns_1:
    X[column]=df[column]
    X_test[column]=test[column]
    X[column].replace(np.inf,np.nan,inplace=True)
    X[column].fillna(X[column].mean(), inplace=True)



# X = X.as_matrix().astype(np.float)
# np.all(np.isfinite(X))

scaler=StandardScaler()
X=scaler.fit_transform(X)
X_test=scaler.fit_transform(X_test)

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


logistic_model=LogisticRegression()
logistic_model.fit(X,y)

xgb_model=xgb.XGBClassifier().fit(X,y)
with open('data/logistic_model_f1+f3.pkl', 'wb') as fid:
    pickle.dump(logistic_model, fid,2)
with open('data/xgb_model_f1+f3.pkl', 'wb') as fid:
    pickle.dump(xgb_model, fid,2)














