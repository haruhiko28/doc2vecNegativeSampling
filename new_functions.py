import numpy as np
import math

def sigm(x):
    sigmoid_range = 34.538776394910684

    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15

    return 1.0 / (1.0 + math.exp(-x))


def PvDw(sentences, window_size):
    # Fill up data batch
    batch_data = []
    label_data = []
    k = 0
    for sentence in sentences:
        while len(sentence) <= window_size:
            sentence.append(0)

        window_sequences = [sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(sentence)]
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]


        batch_and_labels = [(sentence[i:i+window_size], sentence[i+window_size]) for i in range(0, len(sentence)-window_size)]
        batch, labels = [list(x) for x in zip(*batch_and_labels)]
        batch = [x + [k] for x in batch]

        # extract batch and labels
        batch_data.extend(batch)
        label_data.extend(labels)
        k += 1

    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return(batch_data, label_data)



def PvDbow(sentences, window_size):
    batch_data = []
    label_data = []
    k = 0
    for sentence in sentences:

        while len(sentence) <= window_size:
            sentence.append(0)

        window_sequences = [sentence[max((ix-window_size),0):(ix+window_size +1)] for ix, x in enumerate(sentence)]
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]

        batch_and_labels = [(sentence[i:i+window_size], sentence[i+window_size]) for i in range(0, len(sentence)-window_size)]
        batch, labels = [list(x) for x in zip(*batch_and_labels)]
        batch = [x + [k] for x in batch]

        batch_data.extend(batch)
        label_data.extend(labels)
        k += 1

    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return(batch_data[:,window_size],batch_data[:,0:window_size])



def texts_to_embed(embedding,doc_embedding,text_data,text_indices,max_words):
    texts_embed = np.zeros([len(text_data), 200])

    for ix,ts in enumerate(text_data):
        for i in ts:
            texts_embed[ix] += embedding[i]

    texts_embed = texts_embed/max_words
    doc_texts_embed = np.zeros([len(text_indices),  200])

    for ix , j in enumerate(text_indices):
        doc_texts_embed[ix] += doc_embedding[j]

    return np.c_[texts_embed,doc_texts_embed]



def output_changer(label):
    new_label = np.zeros ([len(label),2])

    for ix,_ in enumerate(label):
        if _ == 1:
            new_label[ix,1] = 1
        else:
            new_label[ix,0] = 1
    return new_label
