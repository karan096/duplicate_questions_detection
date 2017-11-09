import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


df= pd.read_csv("data/quora_features.csv")
#print (df.columns)

#Create a new dataFrame
X= pd.DataFrame()

#assign required columns to the data frame
df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]  # .astype(np.float64) ?

X['is_duplicate']=df['is_duplicate']
X['len_q1']=df['len_q1']
X['len_q2']=df['len_q2']
X['diff_len']=df['diff_len']
X['len_char_q1']=df['len_char_q1']
X['len_char_q2']=df['len_char_q2']
X['len_word_q1']=df['len_word_q1']
X['len_word_q2']=df['len_word_q2']
X['common_words']=df['common_words']

test=pd.read_csv("data/quora_features_exp.csv")
#test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]

X_test=pd.DataFrame()
X_test['len_q1']=test['len_q1']
X_test['len_q2']=test['len_q2']
X_test['diff_len']=test['diff_len']
X_test['len_char_q1']=test['len_char_q1']
X_test['len_char_q2']=test['len_char_q2']
X_test['len_word_q1']=test['len_word_q1']
X_test['len_word_q2']=test['len_word_q2']
X_test['common_words']=test['common_words']
X_test['fuzz_qratio']=test['fuzz_qratio']
X_test['fuzz_WRatio']=test['fuzz_WRatio']
X_test['fuzz_partial_ratio']=test['fuzz_partial_ratio']
X_test['fuzz_partial_token_set_ratio']=test['fuzz_partial_token_set_ratio']
X_test['fuzz_partial_token_sort_ratio']=test['fuzz_partial_token_sort_ratio']
# X_test['wmd']=test['wmd']
# X_test['norm_wmd']=test['norm_wmd']
X_test['cosine_distance']=test['cosine_distance']
#
X['fuzz_qratio']=df['fuzz_qratio']
X['fuzz_WRatio']=df['fuzz_WRatio']
X['fuzz_partial_ratio']=df['fuzz_partial_ratio']
X['fuzz_partial_token_set_ratio']=df['fuzz_partial_token_set_ratio']
X['fuzz_partial_token_sort_ratio']=df['fuzz_partial_token_sort_ratio']
# X['wmd']=df['wmd']
# X['norm_wmd']=df['norm_wmd']

X['cosine_distance']=df['cosine_distance']
# X['cityblock_distance']=df['cityblock_distance']
# X['jaccard_distance']=df['jaccard_distance']
# X['canberra_distance']=df['canberra_distance']
# X['euclidean_distance']=df['euclidean_distance']
# X['minkowski_distance']=df['minkowski_distance']
# X['braycurtis_distance']=df['braycurtis_distance']
# X['skew_q1vec']=df['skew_q1vec']
# X['skew_q2vec']=df['skew_q2vec']
# X['kur_q1vec']=df['kur_q1vec']
# X['kur_q2vec']=df['kur_q2vec']

y=X['is_duplicate']

X=X.drop(['is_duplicate'],axis=1)
X = X.as_matrix().astype(np.float)


scaler=StandardScaler()
X=scaler.fit_transform(X)
X_test=scaler.fit_transform(X_test)

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=LogisticRegression()
model.fit(X,y)

print(model.predict_proba(X_test))














