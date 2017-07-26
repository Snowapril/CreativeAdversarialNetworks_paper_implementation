import tensorflow as tf
from layer import generator, Discriminator
from utils import get_noise
import matplotlib.pyplot as plt
import os
import numpy as np

class CAN:
    def __init__(self, sess, z_dim, class_num, path="./model/", name="CAN"):
        self.sess = sess

        # z -> for noise input, x -> for real input, y -> for label of image style
        self.z = tf.placeholder(tf.float32, shape=[None, z_dim])
        self.x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, class_num])
        self.K = class_num

        #model
        with tf.device("/gpu:0"):
            with tf.variable_scope(name+"_model"):
                self.g = generator(self.z)

                self.disc_fake_R, self.disc_fake_C = Discriminator(self.g, self.K)
                self.disc_real_R, self.disc_real_C = Discriminator(self.x, self.K, reuse=True)

            vars = tf.trainable_variables()

            gen_vars = [var for var in vars if "gen" in var.name]
            disc_vars = [var for var in vars if "disc" in var.name]

            #loss function
            self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake_R, labels=tf.zeros_like(self.disc_fake_R))) +\
                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_real_R, labels=tf.ones_like(self.disc_real_R))) +\
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.disc_real_C, labels=self.y))

            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake_R, labels=tf.ones_like(self.disc_fake_R))) +\
                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake_C, labels=tf.ones_like(self.disc_fake_C)/class_num))

            #optimizer
            self.D_opt = tf.train.GradientDescentOptimizer(0.0001).minimize(self.d_loss, var_list=disc_vars)
            self.G_opt = tf.train.AdamOptimizer(0.0001).minimize(self.g_loss, var_list=gen_vars)

        #init and restore
        self.sess.run(tf.global_variables_initializer())
        self.model_path = path
        self.saver = tf.train.Saver()

        if os.path.exists("./model/checkpoint"):
            self.saver.restore(self.sess, self.model_path)
            print("Load Model Complete")
        else:
            print("No Model Found")

    def train(self, x, y, save_model=True, batch_size=128, epochs=100):
        G_loss = []
        Disc_loss = []

        #shuffle data
        shuffle_idx = np.arange(0, x.shape[0])
        np.random.shuffle(shuffle_idx)

        if(not os.path.exists("./model")):
            os.makedirs("./model")

        step_per_epoch = int(x.shape[0]/batch_size)
        print("Train start. num of samples = {}, train epochs = {}, batch_size = {}".format(x.shape[0], epochs, batch_size))
        for epoch in range(epochs):
            train_idx = 0

            for step in range(step_per_epoch):
                batch_xs, batch_ys = x[shuffle_idx[train_idx:train_idx+batch_size]], y[shuffle_idx[train_idx:train_idx+batch_size]]
                sample_z = get_noise(batch_size, 100)

                _, d_cost = self.sess.run([self.D_opt, self.d_loss], feed_dict={self.y:batch_ys, self.x:batch_xs, self.z:sample_z})
                _, g_cost = self.sess.run([self.G_opt, self.g_loss], feed_dict={self.z:sample_z})

                train_idx += batch_size
                if step % 10 == 0:
                    print("Epoch:{}, step:{}, D_cost:{}, G_cost:{}".format(epoch, step, d_cost, g_cost))
                    G_loss.append(g_cost)
                    Disc_loss.append(d_cost)

            if save_model == True:
                save_path = self.saver.save(self.sess, self.model_path)
                print("Epochs:{:01d}, Model saved in file{}".format(epoch, save_path))

            sample_z = get_noise(1, 100)
            generted_image = self.sess.run(self.g, feed_dict={self.z:sample_z})

            fig = plt.figure()
            plt.imshow(generted_image.reshape(256,256,3))
            plt.savefig("./"+"{}.png".format(epoch), bbox_inches="tight")
            plt.close(fig)

        x = np.arange(len(G_loss))

        #train loss graph
        fig = plt.figure()
        plt.plot(x, G_loss, label='Generator')
        plt.plot(x, Disc_loss, label="Discriminator", color="r")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc='lower right')

        fig.savefig("./Loss_graph.png")
        plt.show()

        print("Train model finished")

    def generate(self, z):
        return self.sess.run(self.g, feed_dict={self.z:z})
