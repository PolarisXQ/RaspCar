import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,layers,Model,optimizers
size=[180,240,1]

input=layers.Input(size)
h1=layers.Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(input)
h2=layers.Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(h1)
h3=layers.Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(h2)
h4=layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(h3)
h5=layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(h4)
h6=layers.Flatten()(h5)
h7=layers.Dense(1164, activation='relu')(h6)
h8=layers.Dropout(0.2)(h7)
h9=layers.Dense(100, activation='relu')(h8)
h10=layers.Dropout(0.2)(h9)
h11=layers.Dense(50, activation='relu')(h10)
h12=layers.Dropout(0.2)(h11)
h13=layers.Dense(10, activation='relu')(h12)
h14=layers.Dropout(0.2)(h13)
out=layers.Dense(1, activation='softsign')(h14)
model=Model(inputs=input,outputs=out)
model.compile(optimizer=optimizers.Adam(1e-3), loss='mse')
model.summary()
keras.utils.plot_model(model,show_shapes=True,dpi=200,to_file='nvidia_d2d.png')


# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 180, 240, 1)]     0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 88, 118, 24)       624
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 42, 57, 36)        21636
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 19, 27, 48)        43248
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 17, 25, 64)        27712
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 15, 23, 64)        36928
# _________________________________________________________________
# flatten (Flatten)            (None, 22080)             0
# _________________________________________________________________
# dense (Dense)                (None, 1164)              25702284
# _________________________________________________________________
# dropout (Dropout)            (None, 1164)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               116500
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 50)                5050
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 50)                0
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                510
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 10)                0
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 25,954,503
# Trainable params: 25,954,503
# Non-trainable params: 0
# _________________________________________________________________

