import tensorflow as tf

import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test, y_test, verbose=2)

# Plot das figuras com acurácia
plt.plot(history.history["accuracy"])

plt.title("Acurácia do Modelo")

plt.ylabel("Acurácia")

plt.xlabel("Época")

plt.legend(["Treinamento"], loc="upper left")

plt.show()

# Plot das figuras com perda

plt.plot(history.history["loss"])

plt.title("Perda do Modelo")

plt.ylabel("Perda")

plt.xlabel("Época")

plt.legend(["Treinamento"], loc="upper right")

plt.show()
