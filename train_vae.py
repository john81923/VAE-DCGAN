import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import argparse
import time
import os
import sys
import cPickle
import DataReader
import scipy.misc
from sklearn.manifold import TSNE



#from mnist_data import *
from VAE_cela import ConvVAE
from PIL import Image

'''
vae implementation, alpha version, used with mnist

LOADS of help was taken from:

https://jmetzen.github.io/2015-11-27/vae.html

Ignore this file.
'''


def str2bool(v):
    return v.lower() in("yes","true","t","1")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_epochs', type=int, default=2, #2
                        help='training epochs')
    parser.add_argument('--display_step', type=int, default=10,  #10
                        help='display step')
    parser.add_argument('--checkpoint_step', type=int, default=2,  
                        help='checkpoint step')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--z_dim', type=int, default=512,
                        help='z dim')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--train', type=str2bool, default=False )# training or eval

    parser.add_argument('--indir', type=str )# training or eval

    parser.add_argument('--outdir', type=str )# training or eval


    args = parser.parse_args()
    os.makedirs('repro')
    
    if args.train==False:
        return test(args)
    else:
        return train(args )


def train(args ):

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    display_step = args.display_step
    checkpoint_step = args.checkpoint_step # save training results every check point step
    z_dim = args.z_dim # number of latent variables.
    path =  args.indir #input_file

    if args.train:
        dirname = 'save_train'
    else :
        dirname = 'save'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    vae = ConvVAE(learning_rate=learning_rate, batch_size=batch_size, z_dim = z_dim, train=args.train)

    #mnist = read_data_sets()
    #n_samples = mnist.num_examples

    celabdata = DataReader.DataReader(batch_size=batch_size)
    #path = '/home/master/05/john81923/data/VLDS2018/hw4_dataset/hw4_data'
    train_path = os.path.join(path,'train')
    test_path = os.path.join(path, 'test')
    print train_path
    print test_path
    celabdata.get_data(train_path)
    test_data = celabdata.testdata(test_path)
    n_samples = celabdata.datanumb


     # load previously trained model if appilcable
    ckpt = tf.train.get_checkpoint_state(dirname)
    if ckpt:
     vae.load_model(dirname)

  # Training cycle
    step = 0
    steps=[]
    KLD_fig=[]
    MSE_fig=[]

    print 'ploting fig1_2... please wait.....'
    for epoch in range(training_epochs):
        avg_cost = 0.
        #mnist.shuffle_data()
        train_batch = celabdata.minibatch()
        total_batch = int(n_samples / batch_size)
        print total_batch
        # Loop over all batches
       

        for i in range(total_batch):
            #batch_xs = mnist.next_batch(batch_size)
            batch_xs = train_batch[i]

            # Fit training using batch data
            cost, mse, kl_loss ,new_image,z_log_sigma_sq= vae.partial_fit(batch_xs)

            # Display logs per epoch step
            if i % display_step == 0:
                scipy.misc.imsave('hat.jpg', new_image[0].reshape((64,64,3)))
                steps.append(step)
                KLD_fig.append(kl_loss)
                MSE_fig.append(mse)
            step += 1
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # save model
        if epoch >= 0 and epoch % checkpoint_step == 0:
            checkpoint_path = os.path.join('save', 'model.ckpt')
            vae.save_model(checkpoint_path, epoch)
            print "model saved to {}".format(checkpoint_path)
        

    save_path = 'repro/'
    fig = plt.figure()
    plt.title('KLD')
    plt.plot( steps , KLD_fig )
    plt.savefig( save_path+'tmpfig1_2.jpg',format='png' )

    fig = plt.figure()
    plt.title('MSE')
    plt.plot( steps , MSE_fig )
    plt.savefig( save_path+'tmpfig1_2_.jpg',format='png' )


    pillist= [save_path+'tmpfig1_2.jpg', save_path+'tmpfig1_2_.jpg']


    pilimages = [] # images in each folder
    for file in pillist:
            pilimages.append(Image.open(file))
            w,h = Image.open(file).size
    fig_1_2(pilimages, os.path.join( args.outdir,'fig1_2.jpg'),w,h)

    # save model one last time, under zero label to denote finish.
    #vae.save_model(checkpoint_path, 0)

    return vae

def test(args):

    learning_rate = args.learning_rate
    batch_size = 1 #args.batch_size
    training_epochs = args.training_epochs
    display_step = args.display_step
    checkpoint_step = args.checkpoint_step # save training results every check point step
    z_dim = args.z_dim # number of latent variables.

    

    dirname = 'save'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    vae = ConvVAE(learning_rate=learning_rate, batch_size=batch_size, z_dim = z_dim)

    #mnist = read_data_sets()
    #n_samples = mnist.num_examples

    celabdata = DataReader.DataReader(batch_size=batch_size)
    path = args.indir #input_file
    #train_path = os.path.join(path,'train')
    test_path = os.path.join(path, 'test')
    #print train_path
    print test_path
    #celabdata.get_data(train_path)
    test_data = celabdata.testdata(test_path)
    n_samples = celabdata.datanumb


     # load previously trained model if appilcable
    ckpt = tf.train.get_checkpoint_state(dirname)
    if ckpt:
     vae.load_model(dirname)


    UNIT_SIZE = 64
    target = Image.new('RGB', (UNIT_SIZE*10, UNIT_SIZE*2),255)
    leftone = 0
    lefttwo = 0
    rightone = UNIT_SIZE
    righttwo = UNIT_SIZE

    avg_cost = 0.
    #mnist.shuffle_data()
    #train_batch = celabdata.minibatch()
    #total_batch = int(n_samples / batch_size)
    #print total_batch
    # Loop over all batches
    steps=[]
    KLD_fig=[]
    MSE_fig=[]
    pillist=[]

    for i in range(10):
        #batch_xs = mnist.next_batch(batch_size)
        #batch_xs = train_batch[i]

        # Fit training using batch data
        new_image ,z= vae.testing_fit(test_data[i].reshape((1,64,64,3)))

        scipy.misc.imsave('repro/1_3out{}.jpg'.format(i), new_image[0].reshape((64,64,3)))
        scipy.misc.imsave('repro/1_3in{}.jpg'.format(i), test_data[i].reshape((64,64,3)))     

        pillist.append( 'repro/1_3out{}.jpg'.format(i))
        pillist.append( 'repro/1_3in{}.jpg'.format(i))

    pilimages = [] # images in each folder
    for file in pillist:
            pilimages.append(Image.open(file))
    pinjie(pilimages, os.path.join( args.outdir,'fig1_3.jpg'))

    pillist_14=[]
    for i in range(32):
        #eps = tf.random_normal((10, 512), 0.0, 1.0, dtype=tf.float32)
        z = np.random.uniform(-1, 1, [1, 512]).astype(np.float32)
        new_image  = vae.testing_1_4(z )
        scipy.misc.imsave('repro/1_4out{}.jpg'.format(i), new_image[0].reshape((64,64,3)))
        pillist_14.append('repro/1_4out{}.jpg'.format(i))

    pilimages2 = [] # images in each folder
    for file in pillist_14:
            pilimages2.append(Image.open(file))
    fig_1_4(pilimages2, os.path.join( args.outdir,'fig1_4.jpg'))


    imgdata = celabdata.testdata_tsne(path)
    with open('repro/tsne_z.txt','wb') as f:
        mse_sum = 0
        for i in range(len(imgdata)):
            _ ,mse,z= vae.mse_fit(imgdata[i].reshape((1,64,64,3)))
            mse_sum += mse
            zin = ''
            for t in range(512):
                zin += '%.4f '%(z[0][t]) #"%.2f" % x
            f.write( zin+'\n' )
    print 'test set mse : ', mse_sum
    fdata="repro/tsne_z.txt"
    ftarget="repro/tsne_lb.txt"    
    iris = chj_load_file(fdata,ftarget)

    X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(iris.data)
    #X_pca = PCA().fit_transform(iris.data)
    print("finishe!")
    plt.figure()
    #plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
    plt.savefig( os.path.join( args.outdir,'fig1_5.jpg'),format='png' )   

    return vae

class chj_data(object):
    def __init__(self,data,target):
        self.data=data
        self.target=target

def chj_load_file(fdata,ftarget):
    data=np.loadtxt(fdata, dtype='float32')
    target=np.loadtxt(ftarget, dtype='int32')

    print(data.shape)
    print(target.shape)
    # pexit()

    res=chj_data(data,target)
    return res




def pinjie(images, name ):
    UNIT_SIZE = 64
    target = Image.new('RGB', (UNIT_SIZE*10, UNIT_SIZE*2))   # result is 2*5
    leftone = 0
    lefttwo = 0
    rightone = UNIT_SIZE
    righttwo = UNIT_SIZE
    for i in range(len(images)):
        if(i%2==0):
            target.paste(images[i], (leftone, 0, rightone, UNIT_SIZE))
            leftone += UNIT_SIZE 
            rightone += UNIT_SIZE
        else:
            target.paste(images[i], (lefttwo, UNIT_SIZE, righttwo, UNIT_SIZE*2))
            lefttwo += UNIT_SIZE
            righttwo += UNIT_SIZE
    quality_value = 100
    target.save(name, quality = quality_value)

def fig_1_4(images, name ):
    UNIT_SIZE = 64
    target = Image.new('RGB', (UNIT_SIZE*16, UNIT_SIZE*2))   # result is 2*5
    leftone = 0
    lefttwo = 0
    rightone = UNIT_SIZE
    righttwo = UNIT_SIZE
    for i in range(len(images)):
        if(i%2==0):
            target.paste(images[i], (leftone, 0, rightone, UNIT_SIZE))
            leftone += UNIT_SIZE 
            rightone += UNIT_SIZE
        else:
            target.paste(images[i], (lefttwo, UNIT_SIZE, righttwo, UNIT_SIZE*2))
            lefttwo += UNIT_SIZE
            righttwo += UNIT_SIZE
    quality_value = 100
    target.save( name, quality = quality_value)


def fig_1_2 (images, name,w,h ):
    UNIT_SIZE = w
    target = Image.new('RGB', (w*2, h))   # result is 2*5
    leftone = 0
    rightone = UNIT_SIZE
    for i in range(len(images)):
        target.paste(images[i], (leftone, 0, rightone, h))
        leftone += UNIT_SIZE 
        rightone += UNIT_SIZE
        
    quality_value = 100
    target.save(name, quality = quality_value)


if __name__ == '__main__':
  main()