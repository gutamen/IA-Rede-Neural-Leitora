import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix

# --------------------------------------------------------------------------------------------------------
# PRÉ-PROCESSAMENTO

dataset = pd.read_csv('noticias.csv')
titulos, y = dataset.iloc[:,0], dataset.iloc[:,1]

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

words = [' o ', ' ao ', ' aos ' ,' os ', ' a ', ' as ', ' e ', ' um ', ' uma ', ' ele ', ' ela ', ' eles ', ' elas ',
        ' do ', ' da ', ' dos ', ' das ', ' de ', ' no ', ' na ', ' nos ', ' nas ', ' pelo ',
        ' pela ', ' pelos ', ' pelas ', ' num ', ' numa ', ' nuns ', ' numas ', ' dum ',
        ' duma ', ' duns ', ' dumas ']

filtro = []


# Esvazia palavras inúteis
for title in titulos:
    for word in words:
        title = title.replace(word, ' ')
    filtro.append(title)


vocabulario = ['']
#print(filtro)
for strings in filtro:
    vocabulario += strings.split()

vocabulario = set(vocabulario)

#print(vocabulario)
#vetorizacao = tf.keras.layers.TextVectorization(max_tokens = 10000, output_mode = 'int', output_sequence_length = 30)

#vetorizacao.set_vocabulary(vocabulario)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(vocabulario)
vocab = len(tokenizer.word_docs) + 1

with open('tokenizer_mapping.pickle', 'rb') as handle:
    tokenizerIndex = pickle.load(handle)

tokenizer.word_index = tokenizerIndex


titulos = tokenizer.texts_to_sequences(filtro)

#print(titulos)

max_length = max([len(z) for z in titulos])
titulos = pad_sequences(titulos, maxlen=max_length, padding='post')


model = Sequential()
model.add(Embedding(input_dim=vocab, output_dim=80, input_length=max_length, trainable = True))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.3))
model.add(Dense(units = 7, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.load_weights('modelo.keras')

teste = ["Não vou fazer política do “é dando que se recebe”, diz Lula"]

testeToken = tokenizer.texts_to_sequences(teste)
testeToken = pad_sequences(testeToken, maxlen=max_length, padding='post')

print(testeToken)

#print(titulos)

predicao = model.predict(testeToken)
y_pred_label = np.argmax(predicao, axis=1)

original_class = labelencoder.inverse_transform([y_pred_label])[0]

print(teste)
print(original_class)

