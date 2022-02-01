import multiprocessing
import tensorflow as tf
import glob
import os
size=[180,240]

def make_dataset(img_paths, batch_size,
                 drop_remainder=True, shuffle=True,repeat=1,
                 labels=None,n_map_threads=None,
                 shuffle_buffer_size=None,
                 channels=1,
                 one_hot_depth=3):

    def decode_jpeg(path,label,channels=channels):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=channels)
        if resize:
            img = tf.image.resize(img, resize)
        # img = tf.image.random_crop(img,[resize, resize])
        # img = tf.image.random_flip_left_right(img)
        # img = tf.image.random_flip_up_down(img)
        #img = tf.clip_by_value(img, 0, 255)
        #img=tf.convert_to_tensor(img)
        img = img / 127.5 - 1 #-1~1
        return img,label

    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()

    labels=tf.one_hot(labels,depth=one_hot_depth)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths,labels))

    dataset = dataset.map(decode_jpeg, num_parallel_calls=n_map_threads)

    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)
        # set the minimum buffer size as 2048

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)


    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset= dataset.repeat(repeat)

    len_dataset = len(img_paths) // batch_size

    return dataset , len_dataset

# def label(path = "cannied_data"):
#     files = os.listdir(path)
#     label=[]
#     for file in files:
#         key = int(file[0])
#         label.append(key)
#     return label

# label_list=label()
# batch_size=128
# img_path = glob.glob(r'E:\Python\car\cannied_data\*.jpeg')
# dataset, len_dataset = make_dataset(img_path, batch_size, labels=label_list)
# print(dataset, len_dataset)