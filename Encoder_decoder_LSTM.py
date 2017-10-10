from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from math import sqrt
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

def random_sum_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1, largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y
    
    
def to_string(X, y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        strp = '+'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        Xstr.append(strp)
    max_length = ceil(log10(n_numbers * (largest+1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr
    
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc
    
    
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
           vector = [0 for _ in range(max_int)]
           vector[index] = 1
           pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
           vector = [0 for _ in range(max_int)]
           vector[index] = 1
           pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc
    
def generate_data(n_samples, n_numbers, largest, alphabet):
    X, y = random_sum_pairs(n_samples, n_bnumbers, largest)
    X, y = to_string(X, y, n_numbers, largest)
    X, y = integer_encode(X, y, alphabet)
    X, y = one_hot_encode(X, y, len(alphabet))
    X, y = array(X), array(y)
    return X, y
    
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    striings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)
    
n_terms = 3

largest = 10
alphabet = [str(x) for x in range(10)] + ['+', ' ']

n_chars = len(alphabet)
n_in_seq_length = n_terms * ceil(log10(largest+1)) + n_terms - 1
n_out_seq_length = ceil(log10(n_terms * (largest+1)))

model = Sequential()
model.add(LSTM(75, input_shape=(n_in_seq_length, n_chars))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optiizer='adam', metrics=['accuracy'])
print(model.summary())

X, y = generate_data(75000, n_terms, largest, alphabet)
model.fil(X, y, epochs=1, batch_size=32)

X, y = genrate_data(100, n_terms, largest, alphabet)
loss, acc = model.evaluate(X, y, verbose=0)
print('Loss: %f, Accuracy:%f' % (loss, acc*100))

for _ in range(10):
    X, y = generate_data(1, n_terms, largest, alphabet)
    yhat = model.predict(X, verbose=0)
    in_seq = invert(X[0], alphabet)
    out_seq = invert(y[0], alphabet)
    predicted = invert(yhat[0], alphabet)
    print('%s = %s (expect %s)' % (in_seq, predicted, out_seq))
    
    
