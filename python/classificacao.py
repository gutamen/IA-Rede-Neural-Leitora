# Gustavo Antonio Martini, Gustavo Macedo, Vinicius Gilnek Drage

# --------------------------------------------------------------------------------------------------------
# IMPORTAÇÕES
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

# abre o CSV com os dados
dataset = pd.read_csv('noticias.csv')
titulos, y = dataset.iloc[:,0], dataset.iloc[:,1]

# Cria os labels para cada tipo de notícia
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

# Remove palavras repitidas
for strings in filtro:
    vocabulario += strings.split()

vocabulario = set(vocabulario)

#print(vocabulario)
#vetorizacao = tf.keras.layers.TextVectorization(max_tokens = 10000, output_mode = 'int', output_sequence_length = 30)

#vetorizacao.set_vocabulary(vocabulario)

# Gera o token para mapear as palavras como inteiros
tokenizer = Tokenizer()
tokenizer.fit_on_texts(vocabulario)

vocab = len(tokenizer.word_docs) + 1

titulos = tokenizer.texts_to_sequences(filtro)

#print(titulos)

#Para comparação estabelece qual a maior notícia, em termos de palavras, para o processamento
max_length = max([len(z) for z in titulos])
titulos = pad_sequences(titulos, maxlen=max_length, padding='post')

# Separa o teste e o que será utilizado para o treino, fica 33% para o teste
x_train, x_test, y_train, y_test = train_test_split(titulos, y, test_size=0.33)


# --------------------------------------------------------------------------------------------------------
# REDE NEURAL DENSA

# Criação da rede neural
model = Sequential()
model.add(Embedding(input_dim=vocab, output_dim=80, input_length=max_length, trainable = True))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.3))
model.add(Dense(units = 6, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Ponto de checagem para guarda melhor valor que a rede neural obteve no treinamento
mc = ModelCheckpoint('modelo.kera', monitor='val_acc', save_best_only=True, mode='max')

# Treinamento da rede neural
model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 60, callbacks = [mc])

# Salva o modelo da rede neural em um arquivo para poder ser utilizado postumamente
model.save('modelo.keras')

# Realiza os testes pós treino
model.load_weights('modelo.keras')
print(model.evaluate(x_test, y_test))


#for i in x_test:
#    print(i)

y_pred = model.predict(x_test)

y_pred_labels = np.argmax(y_pred, axis=1)

# Imprime a matriz de confusão, acima fica a coluna que representa cada label
original_class = [labelencoder.inverse_transform([0])[0],labelencoder.inverse_transform([1])[0],labelencoder.inverse_transform([2])[0],labelencoder.inverse_transform([3])[0],labelencoder.inverse_transform([4])[0],labelencoder.inverse_transform([5])[0]]
print(original_class)
matrix = confusion_matrix(y_test, y_pred_labels)

print(matrix)

# Salva os Tokens atribuídos à cada palavra para ser utlizado durante o uso normal da rede neural
with open('tokenizer_mapping.pickle', 'wb') as handle:
    pickle.dump(tokenizer.word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

# --------------------------------------------------------------------------------------------------------



