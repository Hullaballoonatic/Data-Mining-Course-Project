from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from data import get_data

x_train, x_test, y_train, y_test = get_data()

model = Sequential([
    Dense(output_dim=100, input_dim=len(x_train.columns)),  # input layer
    Activation('relu'),  # normalize values to within range -1 to 1
    Dense(output_dim=64),  # hidden layer #1
    Activation('relu'),
    Dense(output_dim=64),  # hidden layer #2
    Activation('relu'),
    Dense(output_dim=10),  # hidden layer #3
    Activation('relu'),
    Dense(output_dim=1),  # output layer
    Activation('softmax')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
# loss is our cost function

model.fit(x_train, y_train, epochs=5, batch_size=32)  # backprop arguments

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)

print(loss_and_metrics)
