import codecs
from os import listdir
from os.path import isfile, join, basename
import os
import gensim
import os
import nltk.data
import numpy
import pathlib
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn import svm, metrics

docLabels = []
folder_name = "Guardian10/Politics"
authors = []

doc_list = []
dir_list = []
file_list = []
i = 10

#gia ka8e arxeio poy vrisketai ston fakelo C10train pare tin dieu8unsi tou arxeiou
for dirpath, dirnames, filenames in os.walk(folder_name):
    #for subdirname in dirnames:
        #authors.append(subdirname)
    for name in filenames:
        file_list.append(pathlib.PurePath(dirpath, name))

#gia ka8e arxeio anoi3e to kai spase to arxeio se protaseis kai topo8etis
for file in file_list:
    f1 = codecs.open(file, "r", "utf-8", errors='ignore')
    st = f1.read()

    authors.append(basename(os.path.abspath(os.path.join(file, os.pardir))))

    #with codecs.open(file, "r",encoding='utf-8', errors='ignore') as st:
    #st = open(str(file),'r').read().decode('unicode_escape').encode('utf-8')
    sent_text = nltk.sent_tokenize(st)
    for sentence in sent_text:
        doc_list.append(sentence)
        docLabels.append(i)
        i=i+1

taggeddoc=[]


for doc, index in zip(doc_list, docLabels):
    td = TaggedDocument(words=nltk.word_tokenize(doc), tags=[str(index)])
    taggeddoc.append(td)

model = gensim.models.Doc2Vec(taggeddoc,  dm = 0, alpha=0.025, size= 10, min_alpha=0.025, min_count=0)


# for epoch in range(10):
#     if epoch % 2 == 0:
#         print ('Now training epoch %s'%epoch)
#     model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.iter)
#     model.alpha -= 0.002  # decrease the learning rate
#     model.min_alpha = model.alpha  # fix the learning rate, no decay
#     model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.iter)


#model.save("doc2vec2.model")
model = Doc2Vec.load("doc2vec.model")

all_vectors=[]

for file in file_list:
    f1 = codecs.open(file, "r", "utf-8", errors='ignore')
    st = f1.read()
    #xorizoume to keimeno se protaseis
    sents = nltk.sent_tokenize(st)

    sentences = []

    for sen in sents:
        doc = nltk.word_tokenize(sen)
        sentences.append(doc)
        #print sentences
        new_vector = model.infer_vector(doc)
    all_vectors.append(new_vector)



# #SVM classification
# result = []
# for i in range(1,10684):
#     string = str(i)
#     result.append(model.docvecs[string])


svc = svm.SVC(kernel='linear', C=1)
svc.fit(all_vectors,authors)


folder_name = "Guardian10/World"
list_of_files =0
#gia ka8e arxeio poy vrisketai ston fakelo C10train pare tin dieu8unsi tou arxeiou
for dirpath, dirnames, filenames in os.walk(folder_name):
    for name in filenames:
        #metraei to plithos twn arxeiwn sto test
        list_of_files += 1
        file_list.append(pathlib.PurePath(dirpath, name))


#gia ka8e arxeio anoi3e to kai spase to arxeio se protaseis kai topo8etis
for file in file_list:
    f1 = codecs.open(file, "r", "utf-8", errors='ignore')
    st = f1.read()
    sent_text = nltk.sent_tokenize(st)
    for sentence in sent_text:
        doc_list.append(sentence)
        docLabels.append(i)
        i=i+1

all_vectors=[]

for file in file_list:
    f1 = codecs.open(file, "r", "utf-8", errors='ignore')
    st = f1.read()
    #xorizoume to keimeno se protaseis
    sents = nltk.sent_tokenize(st)



    sentences = []
    for sen in sents:
        doc = nltk.word_tokenize(sen)
        sentences.append(doc)
        #print sentences
        new_vector = model.infer_vector(doc)
    all_vectors.append(new_vector)

predicted = svc.predict(all_vectors)
#print (predicted)

list_common = []
for a, b in zip(authors, predicted):
    if a == b:
        list_common.append(a)
print (list_common)
print ("Swsta vre8ikan %d apo ta ", len(list_common))

pososto = len(list_common)/list_of_files

print ("%s epitixia", pososto)