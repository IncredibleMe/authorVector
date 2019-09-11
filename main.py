from __future__ import division
from os import listdir
from os.path import isfile, join
import os
import gensim
import os
import nltk.data
import numpy
import pathlib
from nltk import ngrams
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
docLabels = []
folder_name = "C10/C10train"
authors = []

doc_list = []
dir_list = []
file_list = []
i = 10


def get_ngrams(text,n):
    n_grams=ngrams(word_tokenize(text),n)
    return [' '.join(grams) for grams in n_grams]

#gia ka8e arxeio poy vrisketai ston fakelo C10train pare tin dieu8unsi tou arxeiou
for dirpath, dirnames, filenames in os.walk(folder_name):
    for subdirname in dirnames:
        for i in range(1,51):
            authors.append(subdirname)
    for name in filenames:
        file_list.append(pathlib.PurePath(dirpath, name))

#gia ka8e arxeio anoi3e to kai spase to arxeio se protaseis kai topo8etis
for file in file_list:
    st = open(str(file),'r').read()
    sent_text = nltk.sent_tokenize(st)
    for sentence in sent_text:
        doc_list.append(sentence)
        docLabels.append(i)
        i=i+1

taggeddoc=[]



for doc, index in zip(doc_list, docLabels):
    # td = TaggedDocument(words=nltk.word_tokenize(doc), tags=[str(index)])
    td = TaggedDocument(words=get_ngrams(doc,4), tags=[str(index)])
    taggeddoc.append(td)

model = gensim.models.Doc2Vec(taggeddoc,  dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)


for epoch in range(10):
    if epoch % 2 == 0:
        print ('Now training epoch %s'%epoch)
    model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.iter)



model.save("doc2vec.model")

model = Doc2Vec.load("doc2vec.model")

all_vectors=[]

for file in file_list:
    st = open(str(file),'r').read()
    #xorizoume to keimeno se protaseis
    sents = nltk.sent_tokenize(st)

    sentences = []
    for sen in sents:
        doc = nltk.word_tokenize(sen)
        sentences.append(doc)
        #print sentences
        new_vector = model.infer_vector(doc)
    all_vectors.append(new_vector)


# kernels = ['linear', 'rbf', 'poly']
# cs = [0.1, 1, 10, 100, 1000]
# gammas = [0.1, 1, 10, 100]
# #SVM classification
# result = []
# for i in range(1,10684):
#     string = str(i)
#     result.append(model.docvecs[string])

# X_train, X_test, y_train, y_test = train_test_split(
#     all_vectors, authors, test_size=0.2, random_state=42
# )

# for kernel in kernels:
#     for c in cs:
#         for gam in gammas:
#             svc = svm.SVC(kernel=kernel, C=c, gamma=gam).fit(X_train, y_train)
#             print(kernel+" |"+str(c)+" |"+str(gam))
#             print((svc.score(X_test, y_test))*100)


svc = svm.SVC(kernel='rbf', C=1000, gamma=10).fit(all_vectors, authors) #97,6%

# svc = svm.SVC(kernel='linear', C=1000, gamma=10).fit(X_train, y_train)   /// 73,8% akrivia
# svc.fit(all_vectors,authors)
# print((svc.score(X_test, y_test))*100)
# svc.fit(X_train, y_train)

# pred = svc.predict(X_test)
# print((svc.score(X_test, y_test))*100)
# print(confusion_matrix(pred, y_test))


print("-------------------------------------------------")

folder_name = "C10/C10test"

#gia ka8e arxeio poy vrisketai ston fakelo C10train pare tin dieu8unsi tou arxeiou
for dirpath, dirnames, filenames in os.walk(folder_name):
    for name in filenames:
        file_list.append(pathlib.PurePath(dirpath, name))

#gia ka8e arxeio anoi3e to kai spase to arxeio se protaseis kai topo8etis
for file in file_list:
    st = open(str(file),'r').read()
    sent_text = nltk.sent_tokenize(st)
    for sentence in sent_text:
        doc_list.append(sentence)
        docLabels.append(i)
        i=i+1

all_vectors=[]

for file in file_list:
    st = open(str(file),'r').read()
    #xorizoume to keimeno se protaseis
    sents = nltk.sent_tokenize(st)

    sentences = []
    for sen in sents:
        doc = nltk.word_tokenize(sen)
        sentences.append(doc)
        #print sentences
        new_vector = model.infer_vector(doc)
    all_vectors.append(new_vector)

predicted = svc.predict(2)
#print (predicted)

list_common = []
for a, b in zip(authors, predicted):
    if a == b:
        list_common.append(a)
print (list_common)
print ("Swsta vre8ikan "+str(len(list_common))+" apo ta 500" )

pososto = len(list_common)/500

print str(pososto) + " epitixia"