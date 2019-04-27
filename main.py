from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from data import get_data

X_train, Y_train, X_test, Y_test = get_data()

dim = len(X_train.columns)

print('creating model')

model = Sequential([
    Dense(32, input_shape=(dim,)),
    Activation('relu')
])

print('compling model')

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

print('training model')

model.fit(X_train, Y_train, epochs=5, batch_size=32)  # backprop arguments

print('testing model')

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print(loss_and_metrics)
