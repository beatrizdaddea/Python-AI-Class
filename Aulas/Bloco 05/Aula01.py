import tensorflow as tf

# Dataset mnist - 60 mil figuras 28X28 pixels
mnist = tf.keras.datasets.mnist

# Carregar os dados de treino e teste
# x_train= entrada
# y_train = saída
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalização das figuras - (0-255) --> (0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Criar uma rede neural
model = tf.keras.models.Sequential(
    [
        # Entrada tem que ser transformada de figuras 28X28 para um vetor
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # Camada oculta com N neuronios utilizando a função de ativação ReLU
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        # Camada oculta tem X% dos neuronios ativados aleatoriamente
        tf.keras.layers.Dropout(0.2),
        # Camda de sáida - 10 saídas 
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# Definir algoritmo de treinamento, função de perda e a métrica de treinamento
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Treinar a rede
# x_test entrada
# y_test = saída
model.fit(x_train, y_train, epochs=5)

#Avalidar a aacurácia da rede no conjunto de teste
model.evaluate(x_test, y_test, verbose=2)
