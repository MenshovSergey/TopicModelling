import ijson
import pymorphy2
import numpy as np
import lda
import collections

delimiters = ['\n', ' ', ',', '.', '?', '!', ':']
data_file = open("/media/sergej/Elements/data/diplom/posts_all_2016-11-22.txt")
res = ijson.parse(data_file)
docs = []
morph = pymorphy2.MorphAnalyzer()
for prefix, event, value in res:
    if value is not None:
        if prefix.encode("utf-8").endswith("message"):
            if value.encode("utf-8") != "":
                words = value.encode("utf-8").split(" ")
                norm_words = []
                for w in words:
                    norm_words.append(morph.parse(w.decode("utf-8"))[0].normal_form.encode("utf-8"))
                docs.append(norm_words)
    if len(docs) > 100:
        break

import lda.datasets

words = {}
vocab = []
for doc in docs:
    for word in doc:
        if not word in words:
            words[word] = len(words)
            vocab.append(word)

res = np.zeros((len(docs), len(words)),dtype='int64')
for i in range(0, len(docs)):
    c = collections.Counter(docs[i])
    for word,count in c.items():
        res[i][words[word]] = count


model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
np_docs = np.asarray(docs)
model.fit(res)
topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
np.savetxt("a.txt", np_docs, fmt='%s')

print("o")
