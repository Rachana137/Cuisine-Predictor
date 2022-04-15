import pandas as pd
import numpy as np
import json
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


def read_data(path):
    df = pd.read_json(path)
    return df

def convert_string(data, input_ingredients):
    ps = PorterStemmer()
    stem = []
    for ing in data:
        temp = []
        for w in ing:
            w = ps.stem(w)
            temp.append(w)
        stem.append(temp)
    new = []
    for ing in stem:
        x = []
        for w in ing:
            w = w.replace(" ", "-")
            x.append(w)
        new.append(x)
    a = []
    for ii in input_ingredients:
        ii = ii.replace(" ", "-")
        a.append(ii)
    a = ' '.join(a)

    string = []
    for s in new:
        l = ' '.join(s)
        string.append(l)
    string.append(a)
    return string

def document_matrix(x):
    tfidfvectorizer = TfidfVectorizer(min_df=1)
    # pickle.dump(tfidfvectorizer, open('tfvectorizer.pkl', 'wb'))
    tfidf_matrix = tfidfvectorizer.fit_transform(x)

    return tfidf_matrix

def probability_dataframe(value_list, class_list):
    v_list = list()
    c_list = list()
    data = {'value': value_list[0],
            'cuisine': class_list}
    df = pd.DataFrame(data=data)
    return df

def predict_cuisine(df, matrix, ing):
    cuisine = df['cuisine']
    vector = matrix[-1]
    matrix = matrix[:-1]
    # X_train, X_test, y_train, y_test = train_test_split(matrix,cuisine,test_size=0.2,random_state=42)
    model = LogisticRegression(solver='liblinear').fit(matrix, cuisine)
    # model=SVC(gamma = 'auto', probability=True).fit(matrix,cuisine)
    # model=SVC(kernel = 'linear',C=0.01,probability=True).fit(matrix,cuisine)
    predict = model.predict(vector)
    probs = model.predict_proba(vector)
    classes = model.classes_
    dframe = probability_dataframe(probs, classes)
    cuisine_closest = dframe.nlargest(1, 'value')
    closest_score = float(round(cuisine_closest.value, 2))

    return predict, closest_score

def similarity(df, matrix, n):
    vector = matrix[-1]
    matrix = matrix[:-1]
    scores = cosine_similarity(vector, matrix)
    lis = scores.tolist()
    score = list(np.concatenate(lis).flat)
    df['scores'] = score
    score.sort(reverse=True)
    top_n = score[:n]

    dic = {}
    s = []
    for i in range(0, len(top_n), 1):
        for j in range(0, df.index.size):
            if top_n[i] == df["scores"][j]:
                dic = {"id": str(df['id'][j]), "score": round(top_n[i], 2)}
                s.append(dic)
    return s

def output(pred, score, close):
    dictionary = {}
    dictionary['predict'] = pred[0]
    dictionary['score'] = score
    dictionary['closest'] = close
    output = json.dumps(dictionary, indent=4)
    return output




