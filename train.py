import tensorflow as tf
from data import read_image, remove_exception
from model import CAN

def main():
    #K = class number
    image, label, K = read_image()

    sess = tf.Session()
    model = CAN(sess=sess, z_dim=100, class_num=K)

    model.train(image, label, batch_size=15, save_model=False)

if __name__ == "__main__":
    main()
