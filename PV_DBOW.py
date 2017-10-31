# -*- coding: utf-8 -*-

import text_helpers     #the same of Python Cookbook
import new_functions    #This file has original functions

import numpy as np
from sklearn.model_selection import train_test_split
import random
import os

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


data_folder_name = 'temp'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)


vocabulary_size = 7500
embedding_size = 200   # Word embedding size
doc_embedding_size = 200   # Document embedding size
window_size = 3       # How many words to consider to the left.


#downloading data.
print('Loading Data')
texts, target = text_helpers.load_movie_data()


#normalizing texts.
stops = []
print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)


#the length of texts must be longer than window size.
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
texts = [x for x in texts if len(x.split()) > window_size]
assert(len(target)==len(texts))
print('Done.')


# Build our data set and dictionaries.
print('Creating Dictionary')
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_helpers.text_to_numbers(texts, word_dictionary)

# init embeddings and Document embeddings.
embeddings= np.random.uniform(-1,1,[vocabulary_size, embedding_size])
doc_embeddings = np.zeros([len(texts), doc_embedding_size])

#hyperparameter
alpha = 0.05
neg_num = 5
batch_inputs, batch_labels = new_functions.PvDbow(text_data, window_size)

#learning

for ix,y in enumerate(batch_labels):#batch_size繰り返す
    x = batch_inputs[ix]
    neule = np.zeros(doc_embedding_size)
    ns_arr = np.random.randint(0, len(texts), size = neg_num)#negative samplingを作成する
    for i in y:
        Phi = np.dot(doc_embeddings[x],embeddings[i])
        p = new_functions.sigm(Phi)
        g = alpha*(1 - p)
        neule += g*embeddings[i]
        embeddings[i] += g*doc_embeddings[x]

        for ns in ns_arr:#negative sampling
            Phi = np.dot(doc_embeddings[ns],embeddings[i])
            p = new_functions.sigm(Phi.flatten())
            g = alpha * (0 - p)
            neule += g*embeddings[i]
            embeddings[i] += g*doc_embeddings[ns]
    doc_embeddings[x] += neule

np.savez('pvdbow.npz',x = embeddings, y = doc_embeddings)

pvdm = np.load('pvdbow.npz')
embeddings = pvdm['x']
doc_embeddings = pvdm['y']

max_words = 10

#separete train and testdata
train_indices = np.sort(np.random.choice(len(target), round(0.8*len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))

texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]

target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))

text_data_train = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_test]])


X_train = new_functions.texts_to_embed(embeddings,doc_embeddings,text_data_train,train_indices,max_words)
X_test = new_functions.texts_to_embed(embeddings,doc_embeddings,text_data_test,test_indices,max_words)
new_train_label = new_functions.output_changer (target_train)
new_test_label = new_functions.output_changer (target_test)

print(X_train.shape,X_test.shape,new_train_label.shape,new_test_label.shape)

xy = (X_train, X_test, new_train_label, new_test_label)
np.save("./temp/PVDBOW", xy)
