from scipy.misc import imread, imresize
import os
import glob
import numpy as np
import h5py

def im2HDF(path="./wikiart/", resize=True, new_shape=(256,256)):
    #get only 3channel image
    remove_exception()

    img_format = "/*.jpg"
    folder_list = [folder for folder in os.listdir(path) if "." not in folder]

    K = len(folder_list)
    print("Wikiart Class Number:{}".format(K))

    with h5py.File("./store.h5", "w") as store:
        for style in range(K):
            #get image path
            img_list = glob.glob(path + folder_list[style] + img_format)

            if resize:
                img_arr = np.array([imresize(imread(img), size=new_shape) for img in img_list])
            else:
                img_arr = np.array([imread(img) for img in img_list])

            store.create_dataset(folder_list[style], data=img_arr)
            print("write to {} finished".format(folder_list[style]))

    print("write to h5 finished")


def read_image(path="./wikiart/"):
    folder_list = [folder for folder in os.listdir(path) if "." not in folder]
    img_format = "/*.jpg"

    num_img = 0
    for style in folder_list:
        num_img += len(glob.glob(path + style + img_format))

    K = len(folder_list)
    print("Wikiart Class Number:{}".format(K))

    image = np.empty(shape=(num_img, 256, 256, 3))
    label = np.empty(shape=(num_img, K))

    idx = 0

    with h5py.File(path + "store.h5", "r") as store:
        for k in range(K):
            arr = store[folder_list[k]][:]
            num_sample = arr.shape[0]

            image[idx:idx+num_sample] = arr
            label[idx:idx+num_sample] = np.eye(K)[k]

            idx += num_sample

    return image, label, K

def remove_exception(path="./wikiart/"):
    folder_list = [folder for folder in os.listdir("./wikiart") if "." not in folder]

    image_path = np.array([])

    K = len(folder_list)
    for k in range(K):
        img_list = os.listdir(path + folder_list[k])
        img_list = [path + folder_list[k] + "/" + img for img in img_list]

        image_path = np.concatenate((image_path, img_list))

    for path in image_path:
        image = imread(path)
        if image.shape[-1] != 3:
            os.remove(path)
