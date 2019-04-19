from sklearn.model_selection import train_test_split
import tensorflow as tf
import data_loader

# ANN implementation of the model

FILE_NAME = 'ann.h5'  # Save name


def train_network():
    print('Loading data')
    (data, labels) = data_loader.get_data()

    print('Dividing data')
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=20)

    print(x_train[0].shape)

    # This part could be changed
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(256, 256)),
        tf.keras.layers.Dense(120, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=128, epochs=5)
    a = model.evaluate(x_test, y_test)
    print(a)
    save_network(model)


def save_network(model):
    model.save(FILE_NAME)


def load_network():
    model = tf.keras.models.load_model(FILE_NAME)
    model.summary()
    return model


if __name__ == '__main__':
    train_network()
