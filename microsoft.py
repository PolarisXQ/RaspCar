import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,layers,Model,optimizers
size=[180,240,1]
state_input_shape = [1]
activation = 'relu'

pic_input = layers.Input(shape=size)

img_stack = layers.Conv2D(16, (3, 3), name="convolution0", padding='same', activation=activation)(pic_input)
img_stack = layers.MaxPooling2D(pool_size=(2,2))(img_stack)
img_stack = layers.Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1')(img_stack)
img_stack = layers.MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = layers.Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2')(img_stack)
img_stack = layers.MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = layers.Flatten()(img_stack)
img_stack = layers.Dropout(0.2)(img_stack)


#Inject the state input
state_input = layers.Input(shape=state_input_shape)
merged = layers.concatenate([img_stack, state_input])

# Add a few dense layers to finish the model
merged = layers.Dense(64, activation=activation, name='dense0')(merged)
merged = layers.Dropout(0.2)(merged)
merged = layers.Dense(10, activation=activation, name='dense2')(merged)
merged = layers.Dropout(0.2)(merged)
merged = layers.Dense(1, name='output')(merged)

adam = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model = Model(inputs=[pic_input, state_input], outputs=merged)
model.compile(optimizer=adam, loss='mse')
model.summary()
keras.utils.plot_model(model,show_shapes=True,dpi=200,to_file='micrio.png')

# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            [(None, 180, 240, 1) 0
# __________________________________________________________________________________________________
# convolution0 (Conv2D)           (None, 180, 240, 16) 160         input_1[0][0]
# __________________________________________________________________________________________________
# max_pooling2d (MaxPooling2D)    (None, 90, 120, 16)  0           convolution0[0][0]
# __________________________________________________________________________________________________
# convolution1 (Conv2D)           (None, 90, 120, 32)  4640        max_pooling2d[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 45, 60, 32)   0           convolution1[0][0]
# __________________________________________________________________________________________________
# convolution2 (Conv2D)           (None, 45, 60, 32)   9248        max_pooling2d_1[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_2 (MaxPooling2D)  (None, 22, 30, 32)   0           convolution2[0][0]
# __________________________________________________________________________________________________
# flatten (Flatten)               (None, 21120)        0           max_pooling2d_2[0][0]
# __________________________________________________________________________________________________
# dropout (Dropout)               (None, 21120)        0           flatten[0][0]
# __________________________________________________________________________________________________
# input_2 (InputLayer)            [(None, 1)]          0
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 21121)        0           dropout[0][0]
#                                                                  input_2[0][0]
# __________________________________________________________________________________________________
# dense0 (Dense)                  (None, 64)           1351808     concatenate[0][0]
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 64)           0           dense0[0][0]
# __________________________________________________________________________________________________
# dense2 (Dense)                  (None, 10)           650         dropout_1[0][0]
# __________________________________________________________________________________________________
# dropout_2 (Dropout)             (None, 10)           0           dense2[0][0]
# __________________________________________________________________________________________________
# output (Dense)                  (None, 1)            11          dropout_2[0][0]
# ==================================================================================================
# Total params: 1,366,517
# Trainable params: 1,366,517
# Non-trainable params: 0
# __________________________________________________________________________________________________
