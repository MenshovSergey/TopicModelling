# -*- coding: UTF-8 -*-
import lda

import ijson
import nltk
import numpy as np
import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words

delimiters = ['\n', ' ', ',', '.', '?', '!', ':']
data_file = open("/media/sergej/Elements/data/diplom/posts_all_2016-11-22.txt")
res = ijson.parse(data_file)
docs = []
morph = pymorphy2.MorphAnalyzer()


def tokenizerF(t):
    s = t
    s = s.replace(".", " ")
    s = s.replace(",", " ")
    s = s.replace("!", " ")
    s = s.replace("?", " ")
    s = s.replace("–", " ")
    s = s.replace("-", " ")
    s = s.replace("+", " ")
    s = s.replace(")", " ")
    s = s.replace("(", " ")
    s = s.replace("=", " ")
    s = s.replace("&", " ")
    s = s.replace("#", " ")
    s = s.replace(";", " ")
    s = s.replace(":", " ")
    s = s.replace("%", " ")
    s = s.replace("«", " ")
    s = s.replace("»", " ")
    s = s.replace("9", " ")
    s = s.replace("quot", " ")
    for j in range(0, 9):
        s = s.replace(str(j), " ")

    words = nltk.word_tokenize(s, 'russian')
    res = []
    for j in words:
        if len(j) > 2:
            res.append(j)

    return res

user_ids = []
post_ids = []
i = 0
post_id = 0
user_id = 0
for prefix, event, value in res:
    if value is not None:
        if prefix.encode("utf-8").endswith("post_id"):
            post_id = value.encode("utf-8")
        if prefix.encode("utf-8").endswith("user_id"):
            user_id = value.encode("utf-8")
        if prefix.encode("utf-8").endswith("message"):
            if value.encode("utf-8") != "":
                words = tokenizerF(value.encode("utf-8"))
                # print value
                norm_words = []
                for w in words:
                    new = morph.parse(w.decode("utf-8"))[0].normal_form.encode("utf-8")
                    norm_words.append(new)
                docs.append(" ".join(norm_words))
                post_ids.append(post_id)
                user_ids.append(user_id)
            i += 1
    if i > 1000:
        break

import lda.datasets
import pickle

def tokenizerS(t):
    s = t
    s = s.replace(".", " ")
    s = s.replace(",", " ")
    s = s.replace("!", " ")
    s = s.replace("?", " ")
    s = s.replace("–".decode("utf-8"), "")
    s = s.replace("-".decode("utf-8"), "")
    s = s.replace("+".decode("utf-8"), " ")
    s = s.replace(")", " ")
    s = s.replace("(", " ")
    s = s.replace("=", " ")
    s = s.replace("&", " ")
    s = s.replace("#", " ")
    s = s.replace(";", " ")
    s = s.replace(":", " ")
    s = s.replace("%", " ")
    s = s.replace("«".decode("utf-8"), " ")
    s = s.replace("»".decode("utf-8"), " ")
    s = s.replace("9".decode("utf-8"), " ")
    s = s.replace("quot", " ")
    for j in range(0, 9):
        s = s.replace(str(j), " ")

    words = nltk.word_tokenize(s, 'russian')
    res = []
    for j in words:
        if len(j) > 2:
            res.append(j)

    return res


# f = open("user_ids","w")
# pickle.dump(user_ids, f)
# f.close()
# qw = pickle.load(open("post_ids"))


stop_words = get_stop_words('ru')
builder = CountVectorizer(stop_words=stop_words, tokenizer=lambda text: tokenizerS(text))
res = builder.fit_transform(docs)

words = builder.get_feature_names()

# res = coo_matrix((len(docs), len(words)), dtype='int64')

# print builder.get_feature_names()



model = lda.LDA(n_topics=25, n_iter=200, random_state=1)
model.fit(res)
data = model.fit_transform(res)



np.save("data25", data)

# q = np.load("data.npy")

topic_word = model.topic_word_
n_top_words = 8
vocab = np.array(builder.get_feature_names())
for i, topic_dist in enumerate(topic_word):
    ind = np.argsort(topic_dist)[:-n_top_words:-1]
    topic_words = vocab[ind]
    print('Topic {}: {}'.format(i, ' '.join(topic_words).encode("utf-8")))
# np.savetxt("a.txt", res, fmt='%s')

