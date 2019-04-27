from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


def model(df):
    model = Sequential([
        Dense(32, input_size=len(df.columns)),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

    return model
