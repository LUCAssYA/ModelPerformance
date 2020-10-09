from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def NN(shape, n_classes):
    input = Input(shape=shape)
    layer = Dense(512, activation='relu')(input)
    layer = Dense(256, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(32, activation='relu')(layer)
    out = Dense(n_classes, activation='softmax' if n_classes != 1 else 'sigoid')(layer)

    model = Model(input, out)
    model.summary()

    return model
