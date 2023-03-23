import itertools
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


def to_binary_matrix(lst):
    binary = np.zeros((len(lst), 60))
    for i, num in enumerate(lst):
        binary[i][num - 1] = 1
    return binary


# Lendo o arquivo CSV
df = pd.read_csv('data.csv', header=None)

# Convertendo os dados para uma matriz numpy
data = df.values.astype(int)

# Convertendo os dados de entrada em uma matriz bin
binary_data = to_binary_matrix(list(itertools.chain.from_iterable(data)))

# Separando os dados de entrada (X) dos dados de saída (y)
X = binary_data[:-1]
y = binary_data[1:]

# Criando o modelo da rede neural
model = Sequential()
model.add(Dense(128, input_dim=60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# Gerando as próximas possíveis 10 sequências de 6 números
test_data = [data[-1]]
for i in range(20):
    # Convertendo os dados de entrada em uma matriz binária
    binary_test_data = to_binary_matrix(list(itertools.chain.from_iterable(test_data)))
    # Prevendo a próxima sequência de 6 números
    prob = model.predict(binary_test_data[-1].reshape(1, 60))
    next_seq = list(np.argsort(prob[0])[::-1][:6] + 1)
    # Adicionando a próxima sequência à lista de test_data
    test_data.append(next_seq)

# Imprimindo as próximas sequências
for i, seq in enumerate(test_data[1:]):
    print(f"Próxima sequência {i + 1}: {seq}")