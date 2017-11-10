import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = stopwords.words('english')


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return np.divide(v,np.sqrt((v ** 2).sum()))

def common_words(x):
    q1, q2 = x
    q1=q1[:-1]
    q2=q2[:-1]
    return len(set(str(q1).lower().split()) & set(str(q2).lower().split()))

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

data = pd.read_csv('data/quora_duplicate_questions1.tsv',sep=',')
data = data.drop(['id', 'qid1', 'qid2'], axis=1)


data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
data['diff_len'] = data.len_q1 - data.len_q2
data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
data['common_words'] = data[['question1', 'question2']].apply(common_words, axis=1)
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True,
                                                        limit=50000)
data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)


norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True,
                                                             limit=50000)
norm_model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((data.shape[0], 300))

for i, q in tqdm(enumerate(data.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((data.shape[0], 300))

for i, q in tqdm(enumerate(data.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

question1_vectors[np.isnan(question1_vectors)]=0
question2_vectors[np.isnan(question2_vectors)]=0

data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(question1_vectors,
                                                            question2_vectors)]

data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(question1_vectors,
                                                            question2_vectors)]

data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(question1_vectors,
                                                            question2_vectors)]

data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(question1_vectors,
                                                            question2_vectors)]


data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(question1_vectors,
                                                            question2_vectors)]


data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(question1_vectors,
                                                            question2_vectors)]

data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(question1_vectors,
                                                            question2_vectors)]


data['skew_q1vec'] = [skew(x) for x in question1_vectors]
data['skew_q2vec'] = [skew(x) for x in question2_vectors]
data['kur_q1vec'] = [kurtosis(x) for x in question1_vectors]
data['kur_q2vec'] = [kurtosis(x) for x in question2_vectors]
data['cosSim'] = data.apply(lambda x: cosine_sim(x['question1'], x['question2']), axis=1)


#Saving extracted features into a csv
data.to_csv('data/quora_features_exp.csv', index=False)
