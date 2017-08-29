from os import listdir
from os.path import isfile, join
import os
import gensim
import os
import nltk.data
import numpy
import pathlib
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn import svm, metrics

docLabels = []
folder_name = "C10/C10train"

doc_list = []
dir_list = []
file_list = []
i = 10
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

taggeddoc=[]



for doc, index in zip(doc_list, docLabels):
    print (index)
    td = TaggedDocument(words=doc.strip().split(" "), tags=[str(index)])
    taggeddoc.append(td)

# model = gensim.models.Doc2Vec(taggeddoc,  dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)
#
#
# for epoch in range(10):
#     if epoch % 2 == 0:
#         print ('Now training epoch %s'%epoch)
#     model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.iter)
#     model.alpha -= 0.002  # decrease the learning rate
#     model.min_alpha = model.alpha  # fix the learning rate, no decay
#     model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.iter)
#
#
# model.save("doc2vec.model")
model = Doc2Vec.load("doc2vec.model")


file = "C10/C10train/AlanCrosby/10306newsML.txt"
st = open(str(file),'r').read()
doc = st.strip().split(" ")
new_vector = model.infer_vector(doc)


#SVM classification
result = []
for i in range(1,10684):
    string = str(i)
    result.append(model.docvecs[string])


svc = svm.SVC(kernel='linear')
svc.fit(result)


predicted = svc.predict(new_vector)

print("Classification report for classifier %s:\n%s\n"
      % (svc, metrics.classification_report(new_vector, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(new_vector, predicted))
