import os
import pathlib
import re

import gensim
from gensim.models.doc2vec import TaggedDocument
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

id = 0

def get_doc_list(folder_name):
    global id
    doc_list = []
    dir_list = []
    file_list = []
    for dirpath, dirnames, filenames in os.walk(folder_name):
        for name in filenames:
            file_list.append(pathlib.PurePath(dirpath, name))

    for file in file_list:
        st = open(str(file),'r').read()
        doc_list.append(st)
        dir_list.append(str(file).rsplit('/', 2)[-2])
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_list, dir_list


def get_doc(folder_name, type):
    global id
    doc_list, dir_list = get_doc_list(folder_name)
    tokenizer = RegexpTokenizer(r'\w+')

    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()

    taggeddoc = []

    texts = []

    for index,i in enumerate(doc_list):

        # for tagged doc
        wordslist = []
        tagslist = []

        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

        # remove numbers
        number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        number_tokens = ' '.join(number_tokens).split()

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
        # remove empty
        length_tokens = [i for i in stemmed_tokens if len(i) > 1]
        # add tokens to list
        texts.append(length_tokens)

        tagg = str(type)+dir_list[index]+str(id)
        # print (tagg)


        td = TaggedDocument(words=gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),tags=[tagg])
        id = id+1
        if (id==50):
            id = 0
        taggeddoc.append(td)


    return taggeddoc

