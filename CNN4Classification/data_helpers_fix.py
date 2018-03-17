# coding=utf-8
import numpy as np
import itertools
from collections import Counter
import os
import pickle as pkl
import gc
import random

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_text = []
    examples = []
    labels = []
    filedir = '../corpus_fenci'
    for i in os.listdir(filedir):
        print (i)
        example = list(open('../corpus_fenci/' + i, 'r').readlines())
        example = [s.strip() for s in example]
       # print(example)
        x_text.extend(example)  
        #print(x_text)
        examples.append(example)

    # Split by words
    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    i = len(examples)
    l = len(examples)
    for exp in examples:
        label = []
        for j in range(l):
            if j + 1 == i:
                label.append(1)
            else:
                label.append(0)
        i -= 1
        labels.extend([label for _ in exp])

    del examples
    gc.collect()

    #y = np.concatenate(labels, 0)
    return [x_text, labels]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    #max_sequence_length = max(len(x) for x in sentences)
    #print(max_sequence_length)
    sequence_length=40
    print("sequence length is %d"%sequence_length)
    #print (sequence_length)
    padded_sentences = []

    for i in range(len(sentences)):
        if i % 1000 == 0:
            print ("sentence padded:" + str(i))
        sentence = sentences[i]
        if (len(sentence) > sequence_length):
            sentence = sentence[:sequence_length]
            padded_sentences.append(sentence)
        else:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        print(new_sentence,len(new_sentence))
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv.insert(0,"<UNK/>")
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    del word_counts
    gc.collect()

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = [[vocabulary[word] if vocabulary[word]<1200000 else vocabulary['<UNK/>']for word in sentence] for sentence in sentences]
    del sentences
    gc.collect()

    return [x, labels]

def save_pkl(train_data,dev_data,vocabulary,vocabulary_inv) :
    f=open('data_no_sw.pkl','wb')
    pkl.dump({'train_data': train_data,
                'dev_data': dev_data,
                'vocabulary': vocabulary,
                'vocabulary_inv': vocabulary_inv},f)

def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    print("loading data...")
    sentences, labels = load_data_and_labels()
    print('padding sentences...')
    sentences_padded = pad_sentences(sentences)

    del sentences
    gc.collect()

    print('building vocabulary...')
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    print("vocab length is %d"%(len(vocabulary_inv)))
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    print(len(y),len(x))
    del sentences_padded
    gc.collect()

    all_data=list(zip(x,y))
    random.shuffle(all_data)
    train_data=all_data[:int(len(all_data)*0.9)]
    dev_data = all_data[int(len(all_data)*0.9):]
    print("train data size:"+str(len(train_data)))
    print("dev data size:" + str(len(dev_data)))

    print("saving data.pkl")
    if not os.path.exists('data_no_sw.pkl'):
        save_pkl(train_data,dev_data,vocabulary,vocabulary_inv)
    return [train_data, dev_data, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
if __name__=="__main__":
    load_data()
