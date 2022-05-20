import os
from PIL import Image
import tensorflow as tf

def parse(filename, label):
    img = tf.io.read_file(filename)
    dense = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.convert_image_dtype(dense, tf.float32)
    image = tf.image.resize(image, [224, 224])
   
    return image, label


def load_data():    
    f = open("list_attr_celeba.txt", "r")
    #read labels file 
    num_img = int(f.readlines(1)[0].strip())
    attributes = f.readlines(1)[0].split(" ")
    #remove newline from attributes
    del attributes[-1]
    img_paths = []
    img_labels = []
    #1-162770

    #162771-182637

    #182638-202599
    for x in f:
        #process each line in file
        arr = list(filter(None, x.strip().split(" ")))
        path = os.path.join("img_align_celeba/", str(arr[0]))    
        label = arr[1:]
        label = list(map(int, label))
        for i in range(len(label)):
            if label[i] == -1:
                label[i] = 0
        img_paths.append(path)
        img_labels.append(label)
    return img_paths, img_labels

def create_datasets(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE)

    
    train_data = dataset.take(162770)
    temp = dataset.skip(162770)
    val_data = temp.take(182637-162770)
    test_data = temp.skip(202599-182637)

    '''train_data = dataset.take(30)
    temp = dataset.skip(30)
    val_data = temp.take(10)
    test_data = temp.skip(10)'''

    train_data = train_data.batch(64)
    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_data = test_data.batch(64)
    test_data = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_data = val_data.batch(64)
    val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    return train_data, val_data, test_data

def main():
    img_paths, img_labels = load_data()
    train, val, test = create_datasets(img_paths, img_labels)   
    return train, val, test


if __name__ == "__main__":
    main()