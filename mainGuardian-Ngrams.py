from __future__ import division
import codecs
from os import listdir
from os.path import isfile, join, basename
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
import multiprocessing
cores = multiprocessing.cpu_count()

docLabels = []
folder_name = "Guardian10/World"
test_folder = "Guardian10/UK"
authors = []

doc_list = []
dir_list = []
file_list = []
i = 10


def get_ngrams(text,n):
    return [text[i:i+n] for i in range(len(text)-n+1)]


# gia ka8e arxeio poy vrisketai ston fakelo C10train
# pare tin dieu8unsi tou arxeiou
for dirpath, dirnames, filenames in os.walk(folder_name):
    for name in filenames:
        file_list.append(pathlib.PurePath(dirpath, name))

# gia ka8e arxeio apo tin parapanw lista
# anoi3e to kai spase to arxeio se protaseis kai topo8etisetes se lista
for file in file_list:
    f1 = codecs.open(str(file), "r", "utf-8", errors='ignore')
    st = f1.read()

    authors.append(basename(os.path.abspath(os.path.join(str(file), os.pardir))))

    sent_text = nltk.sent_tokenize(st)
    for sentence in sent_text:
        doc_list.append(sentence)
        docLabels.append(i)
        i=i+1

taggeddoc=[]


for doc, index in zip(doc_list, docLabels):
    #print(nltk.word_tokenize(doc))
    td = TaggedDocument(words=nltk.word_tokenize(doc), tags=[str(index)])
    #td = TaggedDocument(words=get_ngrams(doc,4), tags=[str(index)])
    taggeddoc.append(td)

#dimiourgia montelou apo to sunolo protasewn twn keimenwn
model = gensim.models.Doc2Vec(taggeddoc,  dm = 0,window=2, alpha=0.025, vector_size= 100, min_alpha=0.050, min_count=2, workers =cores)


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
    f1 = codecs.open(str(file), "r", "utf-8", errors='ignore')
    st = f1.read()
    #xorizoume to keimeno se protaseis
    sents = nltk.sent_tokenize(st)

    sentences = []

    for sen in sents:
        doc =  get_ngrams(sen, 5)
        sentences.extend(doc)
        new_vector = model.infer_vector(sentences)
    all_vectors.append(new_vector)

# #SVM classification
# result = []
# for i in range(1,10684):
#     string = str(i)
#     result.append(model.docvecs[string])


kernels = ['linear', 'rbf', 'poly']
cs = [0.1, 1, 10, 100, 1000]
gammas = [0.1, 1, 10, 100]





svc = svm.SVC(kernel='linear', C=1000, gamma=10)
svc.fit(all_vectors,authors)

print("----------------------------")


list_of_files =0
file_list = []
#gia ka8e arxeio poy vrisketai ston fakelo C10train pare tin dieu8unsi tou arxeiou
for dirpath, dirnames, filenames in os.walk(test_folder):
    for name in filenames:
        #metraei to plithos twn arxeiwn sto test
        list_of_files += 1
        file_list.append(pathlib.PurePath(dirpath, name))


# #gia ka8e arxeio anoi3e to kai spase to arxeio se protaseis kai topo8etis
# for file in file_list:
#     f1 = codecs.open(str(file), "r", "utf-8", errors='ignore')
#     st = f1.read()
#     sent_text = nltk.sent_tokenize(st)
#     for sentence in sent_text:
#         doc_list.append(sentence)
#         docLabels.append(i)
#         i=i+1

all_vectors=[]
authors = []
for file in file_list:
    f1 = codecs.open(str(file), "r", "utf-8", errors='ignore')
    st = f1.read()
    authors.append(basename(os.path.abspath(os.path.join(str(file), os.pardir))))
    #xorizoume to keimeno se protaseis
    sents = nltk.sent_tokenize(st)


    sentences = []
    for sen in sents:
        #xwrizoume to keimeno se lexeis
        doc =  get_ngrams(sen, 5)
        sentences.extend(doc)
    new_vector = model.infer_vector(sentences)
    #print (sentences)
    all_vectors.append(new_vector)



print ("authors" + str(len(authors)))
#print ("All " + str(len(all_vectors)))
predicted = svc.predict(all_vectors)
print ("predicted" + str(len(predicted)))


list_common = []
for a, b in zip(authors, predicted):
    if a == b:
        list_common.append(a)

print (len(list_common))
print (list_common)
print ("Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(authors)))

pososto = len(list_common)/len(authors)

print ("epitixia : " , pososto)