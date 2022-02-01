import os
import matplotlib.pyplot as plt
import numpy as np
import  glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks
from make_database0 import make_dataset

size = [180, 240, 1]

def label(path = "cannied_data"):
    files = os.listdir(path)
    label=[]
    for file in files:
        key = int(file[0])
        label.append(key)
    return label
def main():
    label_list=label()
    batch_size=128
    img_path=glob.glob(r'E:\Python\car\cannied_data\*.jpeg')
    dataset,len_dataset=make_dataset(img_path,batch_size,labels=label_list,shuffle=False)
    # sample=next(iter(dataset))
    # print(tf.squeeze(sample[0][0],axis=2))
    # plt.imshow(tf.squeeze(sample[0][0],axis=2))
    # plt.show()

    #net
    input=layers.Input(size)
    flatten=layers.Flatten()(input)
    #RNN
    fc1=layers.Dense(32,activation='relu')(flatten)
    fc2=layers.Dense(32,activation='relu')(fc1)
    logits=layers.Dense(3,activation=None)(fc2)

    model1=keras.Model(inputs=input,outputs=logits)
    #keras.utils.plot_model(model1,show_shapes=True,dpi=200,to_file='plot.png')

    checkpoint=callbacks.ModelCheckpoint('checkpoint{epoch}',monitor='val_accuracy',save_best_only=False)
    model1.compile(optimizer=optimizers.Adam(1e-3),loss=losses.CategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    model1.fit(dataset,epochs=10,callbacks=[checkpoint])


if __name__=='__main__':
    main()
