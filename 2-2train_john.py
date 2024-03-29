import os
import time
import argparse
import tensorflow as tf
from scipy.misc import imsave
import dcgan
import DataReader
import numpy as np


class WassersteinGAN(object):
    def __init__(self, g_net, d_net, data, model, scale=10.0):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)

        ddx = tf.gradients(d_hat, x_hat)[0]
        print(ddx.get_shape().as_list())
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        self.d_loss = self.d_loss + ddx

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=64, num_batches=1000000):

        self.sess.run(tf.global_variables_initializer())
        #
        path = '/home/master/05/john81923/data/VLDS2018/hw4_dataset/hw4_data'
        train_path = os.path.join(path,'train')
        data = DataReader.DataReader(batch_size=batch_size)
        data.get_data(train_path)
        #
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            data_batch = data.minibatch()
            #if t % 500 == 0 or t < 25:
            #     d_iters = 100

            for _ in range(0, d_iters):
                bx = data_batch[_]
                bz = batch_z = np.random.uniform( -1, 1, [batch_size, self.z_dim]).astype(np.float32)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim)
            self.sess.run(self.g_adam, feed_dict={self.z: bz, self.x: bx})

            if t % 100 == 0:
                bx = data_batch[0]
                bz = batch_z = np.random.uniform( -1, 1, [batch_size, self.z_dim]).astype(np.float32)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss, g_loss))

            if t % 100 == 0:
                bz = self.z_sampler(batch_size, self.z_dim)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)
                #fig = plt.figure(self.data + '.' + self.model)
                #grid_show(fig, bx, xs.shape)
                bx = self.grid_transform(bx, xs.shape)
                imsave('logs/{}/{}.png'.format(self.data, t/100), bx)
                #fig.savefig('logs/{}/{}.png'.format(self.data, t/100))

    def grid_transform(self, x, size):
        a, b = split(x.shape[0])
        h, w, c = size[0], size[1], size[2]
        x = np.reshape(x, [a, b, h, w, c])
        x = np.transpose(x, [0, 2, 1, 3, 4])
        x = np.reshape(x, [a * h, b * w, c])
        if x.shape[2] == 1:
            x = np.squeeze(x, axis=2)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    #data = importlib.import_module(args.data)
    #model = importlib.import_module(args.data + '.' + args.model)
    # xs = data.DataSampler()
    # zs = data.NoiseSampler()
    d_net = dcgan.Discriminator()
    g_net = dcgan.Generator()
    wgan = WassersteinGAN(g_net, d_net, args.data, args.model)
    wgan.train()