import gensim
import load
# numpy
import numpy

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression


load.get_doc_list('/home/jack/Downloads/C10/C10train')
# documents = load.get_doc('/home/jack/Downloads/C10/C10train')
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

#classifier.fit(train_arrays, train_labels)



