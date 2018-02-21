'''
Created on Feb 11, 2018

@author: Eugene Bulog
Adapted from a predictive character keyboard model used at
https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218
'''


import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams




# setup text read
path = 'inputset.txt'
text = open(path).read().lower()
print('corpus length:', len(text))

# create empty tuples 
word = ()
wordlist = ()

# create list of words separated by spaces, commas, or full stops
for char in text:
    if char.isspace() or char == '.' or char == ',':
        # Add word
        if len(word) > 0:
            wordstr = "".join(word)
            wordlist += (wordstr,)
        # Clear word
        word = ()
    else:
        # add char to word
        word += (char,)

# create sorted list of unique words
word_indices = {}
indices_word = {}
wordlist_sorted = sorted(list(set(wordlist)))

# create dicts for indicies and words
for i in range(len(wordlist_sorted)):
    word_indices[wordlist_sorted[i]] = i
    indices_word[i] = wordlist_sorted[i]

print(len(wordlist_sorted))
print(len(wordlist))

# change sequence length to desired "memory" to check 
# i.e =5 means it will look for patterns within sets of 5 words
SEQUENCE_LENGTH = 5
step = 1
sentences = []
next_word = []
for i in range(0, len(wordlist) - SEQUENCE_LENGTH, step):
    sentences.append(wordlist[i: i + SEQUENCE_LENGTH])
    next_word.append(wordlist[i + SEQUENCE_LENGTH])
print(f'num training examples: {len(sentences)}')

# set up features and labels using one-hot encoding
X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(wordlist_sorted)), dtype=np.bool)
y = np.zeros((len(sentences), len(wordlist_sorted)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_word[i]]] = 1
    
    
# Create model, train, and save. All this can be replaced with the commented loading code if a saved model exists    
    
# Create the model, single LSTM layer with 128 neurons
model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(sentences))))
# fully connected layer with softmax activation for classification
model.add(Dense(len(sentences)))
model.add(Activation('softmax'))  
    
# Default training is 20 epochs with 5% of data used for validation
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history

# save model & history
model.save('keras_model.h5')
pickle.dump(history, open("history.p", "wb"))

#uncomment to load an old model instead of training a new one
#model = load_model('keras_model.h5')
#history = pickle.load(open("history.p", "rb"))


# convert text input into a form usable by the model
def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(wordlist_sorted)))
    
    text = text
    inputword = ()
    inputwordlist = ()
    
    for char in text:
        if char.isspace() or char == '.' or char == ',':
            # Add word
            if len(inputword) > 0:
                wordstr = "".join(inputword)
                inputwordlist += (wordstr,)
            # Clear word
            inputword = ()
        else:
            # add char to word
            inputword += (char.lower(),)
    #encoding
    for t, word in enumerate(inputwordlist):
        x[0, t, word_indices[word]] = 1.
        
    return x

# Get most probable next word
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(1, range(len(preds)), preds.take)


# predict the next word
def predict_completion(text):
    completion = ''
    x = prepare_input(text)
    # get a list of predictions
    preds = model.predict(x, verbose=0)[0]
    
    # get the most likely word's index from these predictions
    next_index = sample(preds)[0]
    
    # get the word found at this index
    next_word = indices_word[next_index]
    completion += next_word
    
    return completion
        
        
# put 5 words present in the input text inside the quotation marks to generate a prediction
print(predict_completion("has gradually become clear to"))