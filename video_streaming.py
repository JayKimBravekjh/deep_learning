from random import random
from random import randint
from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed

def next_frame(last_step, last_frame, column):
    lower = max(0, last_step-1)
    upper = min(last_frame.shape[0]-1, last_step+1)
    step = randint(lower, upper)
    frame = last_frame.copy()
    frame[step, column] = 1
    return frame, step
 
def build_frames(size):
    frames = list()
    frame = zeros((size, size))
    step = randint(0, size-1)
    right = 1 if random() < 0.5 else 0 
    col = 0 if right else size-1
    frame[step, col] = 1
    frames.append(frame)
    
    for i in range(1, size):
       col = i if right else size-1-i 
       frame, step = next_frame(step, frame, col)
       frames.append(frame)
    return frames, right
 
def generate_examples(size, n_patterns):
    X, y = list(), list()
    for _ in range(n_patterns):
        frames, right = build_frames(size)
        X.append(frames)
        y.append(right)
    X = array(X).reshape(n_patterns, size, size, size, 1)
    y = array(y).reshape(n_patterns, 1)
    return X, y
    
size = 50

model = Sequential()
model.add(TimeDistributed(Conv2D(2, (2, 2), activation='relu'), 
     input_shape=(None, size, size, 1)))
 
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

X, y = generate_examples(size, 5000)
model.fit(X, y, batch_size=32, epochs=1)

X, y = generate_examples(size, 100)
loss, acc = model.evaluate(X, y, verbose=0)
print('loss: %f, acc: %f' % (loss, acc*100))

X, y = generate_examples(size, 1)
yhat = model.predict_classes(X, verbose=0)
expected = "Right" if y[0]==1 else "Left"
predicted = "Right" if yhat[0]==1 else "Left"
print('Expected: %s, Predicted: %s' % (expected, predicted))











