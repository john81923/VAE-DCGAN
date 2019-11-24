import os
import scipy.misc
import time
import random

import re
import csv
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np
import cPickle as pk
import random
#from skip_thoughts import skipthoughts


class realimg(object):
    def __init__(self, img, match_sent):
        self.img = img
        self.wimg = []
        self.match_sent = match_sent
        self.mismatch_sent = []
        
    def sent2embed(self, model):
        match_sent = self.match_sent
        if match_sent:
            self.match_embed = skipthoughts.encode(model, match_sent)

        mismatch_sent = self.mismatch_sent
        if mismatch_sent:
            self.mismatch_embed = skipthoughts.encode(model, mismatch_sent)

def get_train_batch(img_objs, batch_size=64):
    data_size = len(img_objs)
    i = tf.train.range_input_producer(data_size, shuffle=False).dequeue()

    img = []
    wimg = []
    match_embed = []
    mismatch_embed = []
    for obj in img_objs:
        img.append(obj.img)
        wimg.append(obj.wimg[0])
        match_embed.append(obj.match_embed[0])
        mismatch_embed.append(obj.mismatch_embed[0])
    img = np.asarray(img)
    wimg = np.asarray(wimg)
    match_embed = np.asarray(match_embed)
    mismatch_embed = np.asarray(mismatch_embed)

    img_tensor = tf.convert_to_tensor(img, name='img_data', dtype=tf.float32)
    img_data = tf.strided_slice(img_tensor, [i, 0, 0, 0], [(i+1), 64, 64, 3])
    img_data = tf.reshape(img_data, [64, 64, 3])
    img_data.set_shape([64, 64, 3])

    wimg_tensor = tf.convert_to_tensor(wimg, name='wrong_img_data', dtype=tf.float32)
    wimg_data = tf.strided_slice(wimg_tensor, [i, 0, 0, 0], [(i+1), 64, 64, 3])
    wimg_data = tf.reshape(wimg_data, [64, 64, 3])
    wimg_data.set_shape([64, 64, 3])

    match_embed_tensor = tf.convert_to_tensor(
        match_embed, name='match_embed_data', dtype=tf.float32)
    match_embed_data = tf.strided_slice(match_embed_tensor, [i, 0], [(i+1), 4800])
    match_embed_data = tf.reshape(match_embed_data, [4800])
    match_embed_data.set_shape([4800])

    mismatch_embed_tensor = tf.convert_to_tensor(
        mismatch_embed, name='mismatch_embed_data', dtype=tf.float32)
    mismatch_embed_data = tf.strided_slice(mismatch_embed_tensor, [i, 0], [(i+1), 4800])
    mismatch_embed_data = tf.reshape(mismatch_embed_data, [4800])
    mismatch_embed_data.set_shape([4800])

    img_batch, wimg_batch, match_embed_batch, mismatch_embed_batch = tf.train.batch(
        [img_data, wimg_data, match_embed_data, mismatch_embed_data], batch_size=batch_size)

    return img_batch, wimg_batch, match_embed_batch, mismatch_embed_batch

def get_test_sent(test_file):
    with open(test_file, "r") as f:
        test_sent = []
        for row in f.read().splitlines():
            test_sent.append(row.split(",")[1])
        model = skipthoughts.load_model()
        vecs = skipthoughts.encode(model, test_sent)
        return vecs
    
def build_imgs():
    path = '/home/master/05/john81923/data/VLDS2018/hw4_dataset/hw4_data'
    with open('../data/tags_clean.csv', 'r') as tag_file:
        tag_reader = csv.reader(tag_file, delimiter='\t')
        img_objs = []
        colors = [
            "red", "orange", "yellow", "green", "blue", "purple", "blonde",
            "pink", "black", "white", "brown"]

        atrribute = ['Bangs','Big_Lips','Black_Hair','Blond_Hair','Brown_Hair','Heavy_Makeup','High_Cheekbones',
            'Male','Mouth_Slightly_Open','Smiling','Straight_Hair','Wavy_Hair','Wearing_Lipstick']
        num = 0
        for row in tag_reader:
            img_id = row[0].split(',')[0]
            tag_row = [row[0].split(',')[1]] + row[1:]
            img = skimage.io.imread(
                '../data/faces/{}.jpg'.format(int(img_id)))
            img = skimage.transform.resize(img, (64, 64))
            match_sent = []
            mismatch_sent = []
            tag_hair = []
            tag_eyes = []
            for tag in tag_row:
                tag = tag.split(':')[0]
                for color in colors:
                    if "{}".format(color) in tag:
                        tag_hair.append(tag)
                    if "{}".format(color) in tag:
                        tag_eyes.append(tag)
            for t_h in tag_hair:
                for t_e in tag_eyes:
                    r = random.random()
                    if r > 0.5:
                        match_sent.append('{} {}'.format(t_h, t_e))
                    else:
                        match_sent.append('{} {}'.format(t_e, t_h))
            if match_sent:
                #  print match_sent
                img_objs.append(realimg(img, match_sent))
                num += 1
                #  if num >= 64: break
                #  print match_sent
        model = skipthoughts.load_model()
        k = 0
        for idx, img_obj1 in enumerate(img_objs):
            find = 0
            for img_obj2 in img_objs[1:]:
                for sent in img_obj2.match_sent:
                    if sent not in img_obj1.match_sent:
                        img_objs[idx].wimg.append(img_obj2.img)
                        img_obj1.mismatch_sent.append(sent)
                        find += 1
                    if find >= 1: break
                if find >= 1: break
            img_obj1.sent2embed(model)
            print "{}/{}".format(k, len(img_objs))
            k += 1

    with open("./train_data/img_objs_new.pk", "w") as f:
        pk.dump(img_objs, f)
#  build_imgs()






class DataReader(object):
    def __init__(self ,batch_size=64):
        self.datanumb = 0
        self.batch_size = batch_size
        self.data_batch =[]
        self.data_pool = []
        

    def get_data(self, path):
        #data_pool = []
        tStart = time.time()
        for oot, dirs, files in os.walk(path,topdown=False):
            dcount = 0
            self.datanumb =  len(files)
            print ('data number ' , len(files))
            for name in files :
                #print name
                dcount+=1
                if dcount%10000==0:
                    print('read file :' , dcount)
                img = scipy.misc.imread( os.path.join(path,name) )
                self.data_pool.append(img/255. )
            break
        tEnd = time.time()
        print ("It cost %f sec" % (tEnd - tStart))
        return self.data_pool

    

    def minibatch(self):
        data_pool = self.data_pool
        random.shuffle( data_pool )
        batch_pool = [data_pool[k:k+self.batch_size] for k in range(0, self.datanumb, self.batch_size)]
        return batch_pool

    def testdata(self,path):
        test_pool = []
        tStart = time.time()
        for oot, dirs, files in os.walk(path,topdown=False):
            dcount = 0
            print ('data number ' , len(files))
            for name in files :
                #print name
                dcount+=1
                if dcount%500==0:
                    print('read file :' , dcount)
                img = scipy.misc.imread( os.path.join(path,name) )
                test_pool.append(img/255.)
            break
        tEnd = time.time()
        print ("It cost %f sec" % (tEnd - tStart))
        return test_pool

    def testdata_tsne(self,path):
        test_pool = []
        tStart = time.time()
        file_name = 'repro/tsne_lb.txt'
        tspath = os.path.join(path,'test')
        with open( os.path.join( path, 'test.csv') , 'r') as tag_file:
            tag_reader = csv.reader(tag_file, delimiter='\t')
            head = True
            clab =[]
            cimg=[]
            fimg=[]
            smile_id = 7
            label_c = np.ones( 1 )
            label_f = np.zeros( 1 )
            count = 0
            with open(file_name, 'wb') as x_file:
                for row in tag_reader:
                    if head:
                        head= False
                        continue
                    img_id = row[0].split(',')[0]
                    img = scipy.misc.imread( os.path.join(tspath,img_id) )/255. 
                    if row[0].split(',')[1:][smile_id] == '1.0':
                        cimg.append(img)
                        x_file.write('{}'.format(1)+'\n')
                    else :
                        cimg.append(img)
                        x_file.write('{}'.format(0)+'\n')
                    if (count+1) %5000 == 0:
                        print count
                    count+=1  
        tEnd = time.time()
        print ("It cost %f sec" % (tEnd - tStart))
        return cimg

    def acgen_data(self):
        atrribute = ['Bangs','Big_Lips','Black_Hair','Blond_Hair','Brown_Hair','Heavy_Makeup','High_Cheekbones',
            'Male','Mouth_Slightly_Open','Smiling','Straight_Hair','Wavy_Hair','Wearing_Lipstick']
        smile_id = 9
        path = '/home/master/05/john81923/data/VLDS2018/hw4_dataset/hw4_data'
        trpath = '/home/master/05/john81923/data/VLDS2018/hw4_dataset/hw4_data/train'
        with open( path+'/train.csv' , 'r') as tag_file:
            tag_reader = csv.reader(tag_file, delimiter='\t')
            head = True
            img_objs =[]
            cimg=[]
            fimg=[]
            label_c = np.ones( 1 )
            label_f = np.zeros( 1 )
            count = 0
            
            for row in tag_reader:
                if head:
                    head= False
                    continue
                img_id = row[0].split(',')[0]
                img = scipy.misc.imread( os.path.join(trpath,img_id) )/255. 
                if row[0].split(',')[1:][smile_id] == '1.0':
                    img_objs.append([img, [1]])
                else :
                    img_objs.append([img, [0]])
                if (count+1) %5000 == 0:
                    print count
                count+=1

        #         if row[0].split(',')[1:][smile_id] == '1.0':
        #             cimg.append(img)
        #         else :
        #             fimg.append(img)
        #         if (count+1) %5000 == 0:
        #             print count
        #         count+=1

        #     if len(cimg)>len(fimg):
        #         small = len(fimg)
        #     else:
        #         small = len(cimg)
        #     print 'number:' ,small

        #     for i in range(small):
        #         ele = []
        #         ele.append(cimg[i])
        #         ele.append(fimg[i])
        #         ele.append(label_c)
        #         ele.append(label_f)
        #         img_objs.append(ele)               

        return img_objs


    def ac_get_batch(self, label_pool ):
        random.shuffle( label_pool )
        
        img_bt_pool = []
        lab_bt_pool = []
        self.datanumb = len(label_pool)
        a_batch = [ label_pool[k:k+self.batch_size] for k in range(0, self.datanumb, self.batch_size)]
        
        for i in range(20):
            img_batch=[]
            lab_batch=[]
            for b in a_batch[i]:
                img_batch.append(b[0])
                lab_batch.append(b[1])
            img_bt_pool.append(img_batch)
            lab_bt_pool.append(lab_batch)
        return img_bt_pool, lab_bt_pool


    def get_train_batch(self,img_objs, batch_size=64):
        data_size = len(img_objs)
        i = tf.train.range_input_producer(data_size, shuffle=False).dequeue()

        img = []
        wimg = []
        match_embed = []
        mismatch_embed = []
        for obj in img_objs:
            img.append(obj[0])
            wimg.append(obj[1])
            match_embed.append(obj[2])
            mismatch_embed.append(obj[3])
        img = np.asarray(img)
        wimg = np.asarray(wimg)
        match_embed = np.asarray(match_embed)
        mismatch_embed = np.asarray(mismatch_embed)

        img_tensor = tf.convert_to_tensor(img, name='img_data', dtype=tf.float32)
        img_data = tf.strided_slice(img_tensor, [i, 0, 0, 0], [(i+1), 64, 64, 3])
        img_data = tf.reshape(img_data, [64, 64, 3])
        img_data.set_shape([64, 64, 3])

        wimg_tensor = tf.convert_to_tensor(wimg, name='wrong_img_data', dtype=tf.float32)
        wimg_data = tf.strided_slice(wimg_tensor, [i, 0, 0, 0], [(i+1), 64, 64, 3])
        wimg_data = tf.reshape(wimg_data, [64, 64, 3])
        wimg_data.set_shape([64, 64, 3])

        match_embed_tensor = tf.convert_to_tensor(
            match_embed, name='match_embed_data', dtype=tf.float32)
        match_embed_data = tf.strided_slice(match_embed_tensor, [i, 0], [(i+1), 10])
        match_embed_data = tf.reshape(match_embed_data, [10])
        match_embed_data.set_shape([10])

        mismatch_embed_tensor = tf.convert_to_tensor(
            mismatch_embed, name='mismatch_embed_data', dtype=tf.float32)
        mismatch_embed_data = tf.strided_slice(mismatch_embed_tensor, [i, 0], [(i+1), 10])
        mismatch_embed_data = tf.reshape(mismatch_embed_data, [10])
        mismatch_embed_data.set_shape([10])

        img_batch, wimg_batch, match_embed_batch, mismatch_embed_batch = tf.train.batch(
            [img_data, wimg_data, match_embed_data, mismatch_embed_data], batch_size=batch_size)

        return img_batch, wimg_batch, match_embed_batch, mismatch_embed_batch


if __name__=='__main__':
    path = '/home/master/05/john81923/data/VLDS2018/hw4_dataset/hw4_data'
    train_path = os.path.join(path,'train')
    print train_path
    dataC = DataReader()
    img_objs = dataC.acgen_data()
    img_batch, lab_batch = dataC.ac_get_batch(img_objs)

    #img_batch, wimg_batch, match_embed_batch, mismatch_embed_batch= dataC.get_train_batch(img_objs)

    # data_pool = dataC.minibatch(train_path)
    # scipy.misc.imsave('tcelA.jpg', data_pool[20])
    # scipy.misc.imsave('tcelB.jpg', data_pool[30])



        
