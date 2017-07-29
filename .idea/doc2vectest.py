import gensim
import load
# numpy
import numpy

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression


#load.get_doc_list('/home/jack/Downloads/C10/C10train')
documents = load.get_doc('/home/jack/Downloads/C10/C10train', 'TRAIN')
documents += load.get_doc('/home/jack/Downloads/C10/C10test', 'TEST')
# documents += load.get_doc('/home/jack/Downloads/C10/C10test/AlexanderSmith')
print ('Data Loading finished')
print (len(documents),type(documents))

#print (len(documents))


# model = gensim.models.Doc2Vec(documents, dm = 0, alpha=0.025, size= 300, min_alpha=0.025, min_count=0)
model = gensim.models.Doc2Vec(documents, size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)


print (model.docvecs.count)


for epoch in range(10):
    if epoch % 20 == 0:
        print ('Now training epoch %s'%epoch)
    model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    model.train(documents, total_examples=model.corpus_count, epochs=model.iter)



classifier = LogisticRegression()

authors=['DavidLawder', 'RobinSidel', 'MureDickie', 'AlanCrosby', 'JaneMacartney', 'JimGilchrist', 'AlexanderSmith', 'MarcelMichelson', 'ToddNissen', 'BenjaminKangLim']

train_arrays = numpy.zeros((500, 300))
# train_arrays = []
train_labels = [''] * 500

for i in range(10):
    for j in range(50):
        prefix_train = 'TRAIN' + authors[i] + str(j)
        train_arrays[i] = model.docvecs[prefix_train]
        train_labels[i] = authors[i]

test_arrays = numpy.zeros((500, 300))
#test_labels = numpy.zeros(500)
test_labels = [''] * 500

for i in range(10):
    for j in range(50):
        prefix_test = 'TEST' + authors[i] + str(j)
        test_arrays[i] = model.docvecs[prefix_test]
        test_labels[i] = authors[i]

classifier.fit(train_arrays, train_labels)


print (classifier.score(test_arrays, test_labels))