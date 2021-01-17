#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:51:50 2019

@author: yingwang
"""

import numpy as np
import pandas as pd
from keras.models import Model,load_model
from keras import backend as K
from collections import defaultdict


def list2data(list_file, kmer_data, k):
    file_number = len(list_file)
    data = np.zeros((file_number, 4**k),dtype='float32')
    for i in range(len(list_file)):
        data[i] = (kmer_data[list_file[i]])
    return data

# 计算loss值，x为anchor，y为positive，z为negtive
def eulidean_distance(vects):
    x,y,z = vects
    sum_square = K.sqrt(K.sum(K.square(x-y),axis=1,keepdims=True))-K.sqrt(K.sum(K.square(x-z),axis=1,keepdims=True))+0.5
   # sum_square = K.sqrt(K.sum(K.square(x-y),axis=1,keepdims=True))-K.sqrt(K.sum(K.square(x-z),axis=1,keepdims=True))
    return K.maximum(sum_square,0)

def identity_loss(y_true,y_pred):   
    return K.mean(y_pred-0*y_true)

def list_duplicates(seq):    
    tally = defaultdict(list)    
    for i,item in enumerate(seq):        
        tally[item].append(i)    
    return ((key,locs) for key,locs in tally.items() if len(locs)>=1) 

def get_data(file_name):
    data = {}
    for i in range(len(file_name)):
         m =pd.read_csv('kmc6_466_307/'+file_name[i]+'_k6.txt', header=None,sep='\t',index_col=[0]).T
         new_data = (m.reindex(columns=AT, fill_value=0)).iloc[0]
     #   data[file_name[i]] =np.around((new_data)*10000,6)
         data[file_name[i]] =np.around((new_data/new_data.sum())*10000,6)
    return data


if __name__ == "__main__":

    train_data = pd.read_csv('new_data_filter.csv')
    all_species = list(train_data.iloc[:, -1])
    all_name_ = ['_'.join(['_'.join(_.split())]) for _ in all_species]

    vali_name = list(pd.read_table('vali_name_1.txt',header = None,index_col=0).T)
    train_name = list(set(all_name_).difference(set(vali_name)))
    phylum_ = list(train_data['phylum'])
    class_ = list(train_data['class'])
    order_ = list(train_data['order'])
    family_ = list(train_data['family'])
    genus_ = list(train_data['genus'])
    AT = list(pd.read_table('k'+str(6)+'.txt', header=None, index_col=0).T)
    data = get_data(all_name_)

    test_category = {}
    for test_iterm in vali_name:
        test_category[test_iterm] = genus_[all_name_.index(test_iterm)]

    order = []
    family = []
    genus = []
    all_name = []
    cc = list(zip(order_,family_,genus_,all_name_))
    mid = []
    for test_iterm in train_name:
        mid.append(cc[all_name_.index(test_iterm)])
    order[:],family[:],genus[:],all_name[:]= zip(*mid)
    

    non_unique_train = []
    for dup in sorted(list_duplicates(genus)):
        non_unique_train.append(dup)



    model1 = load_model('good_model/29.h5',custom_objects={'identity_loss':identity_loss,'eulidean_distance':eulidean_distance})
    model = Model(inputs=model1.input[0],outputs=model1.get_layer('Dense_20').get_output_at(0))


    g = 0
    f = 0
    o = 0
    c =0
    p = 0
    n = 0
    for anchor_spiece in vali_name:

        all_ = {}
        for positive_iterm in non_unique_train: 
       
            name = positive_iterm[0]
            test_genus = [anchor_spiece]
            for i in positive_iterm[1]:
                test_genus.append(all_name[i])
            input_data = list2data(test_genus,data,6)
            output_data = model.predict(input_data)
            oushi = []
            for j in output_data[1:]:
                oushi.append(np.sqrt(sum(np.power((output_data[0] - j), 2))))
            
            all_[name] = np.mean(oushi)
        
            
        all_ = sorted(all_.items(), key=lambda x: x[1])

            
    
        if all_[0][0] == test_category[anchor_spiece]:
            g = g +1
        if family_[genus_.index(all_[0][0])] == family_[all_name_.index(anchor_spiece)]:
            f = f+1
        if order_[genus_.index(all_[0][0])] == order_[all_name_.index(anchor_spiece)]:
            o = o+1                
        if class_[genus_.index(all_[0][0])] == class_[all_name_.index(anchor_spiece)]:
            c = c+1
        if phylum_[genus_.index(all_[0][0])] == phylum_[all_name_.index(anchor_spiece)]:
            p = p+1
    n = len(vali_name)        
    print(g,f,o,c,p)
    print(n)
    print(g/n,f/n,o/n,c/n,p/n)

              
                    
    
