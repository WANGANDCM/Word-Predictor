# Word-Predictor
A simple recurrent neural net that predicts the next word in a sentence based on the previous words in the sentence. Built with Tensorflow &amp; Keras. 

Adapted from a character-based predictive keyboard example at https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218


A model which has been pre-trained on the included .txt example set is included in this repo for reference. The training set is the same one used in the example in the above link.

This network is very data hungry, so for best performance and accuracy, if you are training with new data I would reccommend using a very large data set and the default 20 epochs, however this may take several hours to train.
