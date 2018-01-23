import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.model_selection import train_test_split

BASE_DIR = 'C:\\Users\sorablaze_11\Desktop\LSTM - pro_ad\data_sets\\'

data = pd.read_csv(BASE_DIR + 'text_emotion.csv')
data = data[['content', 'sentiment']]

emotions = list(set(data['sentiment']))
#for i in emotions:
#    print(i, ' ', len(data[data.sentiment == i]))

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['content'].values)
X = tokenizer.texts_to_sequences(data['content'].values)
X = pad_sequences(X)

embed_dim = 128
#print(X.shape)

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(len(emotions), activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
#print(model.summary())
#print(model.input_shape)
#print(model.output_shape)

Y = pd.get_dummies(data['sentiment'].values)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
#print(X_train.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape)
#print(Y_train)

batch_size = 32
model.fit(np.array(X_train), np.array(Y_train), epochs = 7, batch_size = batch_size, verbose = 1)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(np.array(X_test), np.array(Y_test), verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

temp = X[2]
temp = np.reshape(temp, (1, 35))
#temp.shape
#Y

emotions[np.argmax(model.predict(temp))]
