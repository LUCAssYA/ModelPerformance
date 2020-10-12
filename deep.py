from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential


def NN(shape, n_classes):

    input = Input(shape=shape)

    layer = Dense(128,  activation = 'relu')(input)
    layer = Dense(64, activation = 'relu')(layer)
    layer = Dense(32, activation = 'relu')(layer)
    if n_classes == 1:
        out = Dense(n_classes, activation='sigmoid')(layer)
    else:
        out = Dense(n_classes, activation='softmax')(layer)

    model = Model(input, out)

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

    return model
