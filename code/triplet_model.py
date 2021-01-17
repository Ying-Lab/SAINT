# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:14:00 2019

@author: wangyiwen
"""
#import shap
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import os
import keras
os.environ['KERAS_BACKEND']='tensorflow'



from keras.models import Model
from keras.layers import Input, Dense, Lambda,Dropout
from keras import backend as K
from keras.callbacks import Callback

from collections import OrderedDict
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
np.random.seed(123) 

from sklearn.utils.class_weight import compute_class_weight

def list_duplicates(seq,index):    
    tally = defaultdict(list) 
    for i,item in enumerate(seq): 
        if i in index and item != '':
            tally[item].append(i)   
    return ((key,locs) for key,locs in tally.items() if len(locs)>1) 

def list_duplicates1(seq):    
    tally = defaultdict(list) 
    for i,item in enumerate(seq): 
        tally[item].append(i)   
    return ((key,locs) for key,locs in tally.items() if len(locs) >= 1) 



def triplet(level,all_level,file_name,index):
    anchor = []
    positive = []
    negative = []
    non_unique_train = []
    new_index = []
 
    for dup in sorted(list_duplicates(all_level[0],index)):
        if level != all_level[0]:
            new_index.extend( random.sample(dup[1], 1))
        else:
            new_index = index
    label = []  
    for dup in sorted(list_duplicates(level,new_index)):    
        non_unique_train.append(dup)
    for iterm in non_unique_train:
        for anchor_iterm in iterm[1]:
            positive_index = iterm[1][:]
            positive_index.remove(anchor_iterm)
            for positive_iterm in positive_index:
                negative_index =   list( (set(new_index) | set(iterm[1])) - (set(new_index) & set(iterm[1])))                
                for negative_iterm in negative_index:
                    if level == all_level[3]:
                        if all_level[2][anchor_iterm] != all_level[2][positive_iterm] and all_level[4][positive_iterm] == all_level[4][negative_iterm] != '':
                            anchor.append(file_name[anchor_iterm])
                            positive.append(file_name[positive_iterm])
                            negative.append(file_name[negative_iterm])
                            for i in range(6):
                                if all_level[i][anchor_iterm] == all_level[i][positive_iterm]:
                                    same_po = i+1
                                    break
                            for i in range(6):
                                 if all_level[i][anchor_iterm] == all_level[i][negative_iterm]:
                                      different_po = i+1
                                      break
                            label.append(different_po-same_po)
                    else:
                            anchor.append(file_name[anchor_iterm])
                            positive.append(file_name[positive_iterm])
                            negative.append(file_name[negative_iterm])
             
                            for i in range(6):
                                if all_level[i][anchor_iterm] == all_level[i][positive_iterm]:
                                    same_po = i+1
                                    break
                            for i in range(6):
                                 if all_level[i][anchor_iterm] == all_level[i][negative_iterm]:
                                      different_po = i+1
                                      break
                 
                            label.append(different_po-same_po)
      
    return anchor,positive,negative,label

def triplet1(level,all_level,file_name,index):
    anchor = []
    positive = []
    negative = []
    non_unique_train = []
    new_index = []
 
    for dup in sorted(list_duplicates(all_level[0],index)):
        if level != all_level[0]:
            new_index.extend( random.sample(dup[1], 1))
        else:
            new_index = index
    label = []  
    for dup in sorted(list_duplicates(level,new_index)):    
        non_unique_train.append(dup)
    for iterm in non_unique_train:
        for anchor_iterm in iterm[1]:
            positive_index = iterm[1][:]
            positive_index.remove(anchor_iterm)
            for positive_iterm in positive_index:
                negative_index =   list( (set(new_index) | set(iterm[1])) - (set(new_index) & set(iterm[1])))                
                for negative_iterm in negative_index:

                        if all_level[2][anchor_iterm] != all_level[2][positive_iterm] and all_level[4][positive_iterm] == all_level[4][negative_iterm] != '':
                            anchor.append(file_name[anchor_iterm])
                            positive.append(file_name[positive_iterm])
                            negative.append(file_name[negative_iterm])
                            for i in range(6):
                                if all_level[i][anchor_iterm] == all_level[i][positive_iterm]:
                                    same_po = i+1
                                    break
                            for i in range(6):
                                 if all_level[i][anchor_iterm] == all_level[i][negative_iterm]:
                                      different_po = i+1
                                      break
                            label.append(different_po-same_po)

      
    return anchor,positive,negative,label


def list2data1(list_file, kmer_data, k):
    data = []
    for i in range(len(list_file)):
        kk = kmer_data[list_file[i]]
        data.append(list(kk))
    return data

def list2data(list_file, kmer_data, k):
    file_number = len(list_file)
    data = np.zeros((file_number, 4**6),dtype='float32')
    for i in range(len(list_file)):
        data[i] = (kmer_data[list_file[i]])
    return data


#随机打乱顺序
def shuffle(a,b,c,d): 
     multi_list = list(zip(a, b, c,d))

     random.shuffle(multi_list)     
     a[:], b[:],c[:] ,d[:]=  zip(*multi_list) 
     return a,b,c,d

# 计算loss值，x为anchor，y为positive，z为negtive
def eulidean_distance(vects):
    x,y,z = vects
    sum_square = K.sqrt(K.sum(K.square(x-y),axis=1,keepdims=True))-K.sqrt(K.sum(K.square(x-z),axis=1,keepdims=True))+0.5
    return K.maximum(sum_square,0)

def identity_loss(y_true,y_pred):   
    return K.mean(y_pred-0*y_true)


def triplet_model():
    query_input = Input(shape=(4**6, ), name='Query_Input', dtype='float32')
    positive_input = Input(shape=(4**6,), name='Positive_Input', dtype='float32')
    negative_input = Input(shape=(4**6,), name='Negative_Input', dtype='float32')


    dense1 = Dense(200, activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_1")                                                                                               
    dense2 = Dense(200, activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_2")                                                                                               
    dense3 = Dense(200, activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_3")                                                                                               
    dense4 = Dense(200, activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_4")                                                                                              
    dense5 = Dense(200,activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_5")                                                                                                
    dense6 = Dense(200,activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_6")                                                                                              
    dense7 = Dense(200,activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_7")                                                                                              
    dense8 = Dense(100,activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_8")
    dense20 = Dense(100, use_bias=True, kernel_initializer="glorot_normal",name="Dense_20")

    que_dense1 = dense1(query_input)
    pos_dense1 = dense1(positive_input)
    neg_dense1 = dense1(negative_input)
   
    que_dense2 = dense2(que_dense1)
    pos_dense2 = dense2(pos_dense1)
    neg_dense2 = dense2(neg_dense1)
    
    que_dense3 = dense3(que_dense2)
    pos_dense3 = dense3(pos_dense2)
    neg_dense3 = dense3(neg_dense2)
        
    que_dense4 = dense4(que_dense3)
    pos_dense4 = dense4(pos_dense3)
    neg_dense4 = dense4(neg_dense3)
   
         
    que_dense5 = dense5(que_dense4)
    pos_dense5 = dense5(pos_dense4)
    neg_dense5 = dense5(neg_dense4)
    
    que_dense6 = dense6(que_dense5)
    pos_dense6 = dense6(pos_dense5)
    neg_dense6 = dense6(neg_dense5)
    
    que_dense7 = dense7(que_dense6)
    pos_dense7 = dense7(pos_dense6)
    neg_dense7 = dense7(neg_dense6)
    
    que_dense8 = dense8(que_dense7)
    pos_dense8 = dense8(pos_dense7)
    neg_dense8 = dense8(neg_dense7)

    que_out = dense20(que_dense8)
    pos_out = dense20(pos_dense8)
    neg_out = dense20(neg_dense8)
    

    
    triplet_loss = Lambda(eulidean_distance, 
                  output_shape = (1,))([que_out,pos_out,neg_out])
    model = Model(inputs = [query_input, positive_input, negative_input],outputs = triplet_loss)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=identity_loss, optimizer=adam)
    return model

def predict(anchor_list,positive_list,negative_list,model1,model2,model3,model,is_evalute):    
    p = 0
    a = b = c = np.zeros((1,100)) 
    score = []           
    while p <=int(len(anchor_list)/100000): 
        list_batch = list(anchor_list[100000*p:min((p+1)*100000,len(anchor_list))]) + list(positive_list[100000*p:min((p+1)*100000,len(anchor_list))]) +    \
                               list(negative_list[100000*p:min((p+1)*100000,len(anchor_list))])
        list_batch = (list(set(list_batch)))
        data = get_data(list_batch)
        
        anchor_data = list2data(anchor_list[100000*p:min((p+1)*100000,len(anchor_list))],data,k)
        positive_data = list2data(positive_list[100000*p:min((p+1)*100000,len(anchor_list))], data, k)
        negative_data = list2data(negative_list[100000*p:min((p+1)*100000,len(anchor_list))], data, k)

        a_ =  model1.predict([anchor_data,positive_data, negative_data])
        b_ =  model2.predict([anchor_data,positive_data, negative_data])
        c_ =  model3.predict([anchor_data,positive_data, negative_data]) 
        if is_evalute:
             score.append(model.evaluate([anchor_data,positive_data, negative_data], np.zeros(len(anchor_data)),batch_size=5120,verbose = 1))
        a = np.concatenate([a,a_],axis=0)
        b = np.concatenate([b,b_],axis=0)
        c = np.concatenate([c,c_],axis=0)                
        p=p+1
    a = np.delete(a,0,axis=0)
    b = np.delete(b,0,axis=0)
    c = np.delete(c,0,axis=0)
    list_batch = data =anchor_data = positive_data = negative_data = a_ = b_ = c_ = []    
    return a,b,c,np.mean(score)

def plt_triplet(a,b,c):
    same=[]
    different=[]
    for x in range(len(a)):
        same.append(np.sqrt(np.sum(np.square(np.array(a[x])-np.array(b[x])))))
        different.append(np.sqrt(np.sum(np.square(np.array(a[x])-np.array(c[x]))))) 
    return same,different

def train_test_triplet(kingdom,phylum,class_,order,family,genus,file_name,index,is_):
    anchor_list_class,positive_list_class,negative_list_class,label_class = triplet(class_,[genus,family,order,class_,phylum,kingdom],file_name,index)
  
    anchor_list_order,positive_list_order,negative_list_order,label_order = triplet(order,[genus,family,order,class_,phylum,kingdom],file_name,index)
    anchor_list_family,positive_list_family,negative_list_family,label_family = triplet(family,[genus,family,order,class_,phylum,kingdom],file_name,index)
    anchor_list_genus,positive_list_genus,negative_list_genus,label_genus = triplet(genus,[genus,family,order,class_,phylum,kingdom],file_name,index)
 
    print(len(anchor_list_class))
    print(len(anchor_list_order))
    print(len(anchor_list_family))
    print(len(anchor_list_genus))
    
    

    anchor_list = anchor_list_class + anchor_list_order + anchor_list_family + anchor_list_genus
    positive_list = positive_list_class + positive_list_order + positive_list_family + positive_list_genus
    negative_list = negative_list_class + negative_list_order + negative_list_family + negative_list_genus
    distance_label = label_class + label_order + label_family + label_genus
    category_label = ['class']*len(anchor_list_class)+['order']*len(positive_list_order)+['family']*len(positive_list_family)+['genus']*len(positive_list_genus)



    return anchor_list,positive_list,negative_list,category_label,distance_label

def get_data(file_name):
    data = {}
    for i in range(len(file_name)):
        m =pd.read_csv('kmer/'+file_name[i]+'_k6.txt', header=None,sep='\t',index_col=[0]).T
          
        new_data = (m.reindex(columns=AT, fill_value=0)).iloc[0]
     #   data[file_name[i]] =np.around((new_data)*10000,6)
        data[file_name[i]] =np.around((new_data/new_data.sum())*10000,4)
      #  print(data)
        
    return data


def weight_distance(weight,distance_label):
     new_label = []
     for i in distance_label:
         if i == 1:
             new_label.append(weight[i-1]*16)
     for i in distance_label:
         if i == 2:
             new_label.append(weight[i-1]*8)
     for i in distance_label:
         if i == 3:
             new_label.append(weight[i-1]*4)
     for i in distance_label:
         if i == 4:
             new_label.append(weight[i-1]*2)
     for i in distance_label:
         if i == 5:
             new_label.append(weight[i-1]*1)
     return new_label


class callbackmodel(Callback):
    def __init__(self, model):
        self.model = model
    def on_epoch_end(self,epoch,logs=None):


        self.model.save('good_model/'+str(epoch)+'.h5')


#训练模型
def train(kingdom,phylum,class_,order,family,genus,file_name,l,k):
    print('--------On_train_begin-----------')
    
    
    vali_name =  list(pd.read_table('test.txt', header=None, index_col=0).T)
   
    train_name = list(set(file_name).difference(set(vali_name)))
    vali_index = []
    train_index = []

    for i in vali_name:
        vali_index.append(file_name.index(i))
    for j in train_name:
        train_index.append(file_name.index(j))
  
    model = triplet_model()
  
 
    '''   训练集  '''
    anchor_list_train, positive_list_train,negative_list_train,category_label,distance_label = train_test_triplet(kingdom,phylum,class_,order,family,genus,file_name,train_index,False)                
    anchor_list_train, positive_list_train, negative_list_train,distance_label=shuffle(anchor_list_train,positive_list_train,negative_list_train,distance_label)

    print('train',len(anchor_list_train))
    print(len(category_label),len(distance_label))


    weight = compute_class_weight('balanced', [1,2,3,4,5], distance_label)
    
    last_weight = weight_distance(weight,distance_label)       

   
    ''' 验证集   '''
    anchor_list_vali, positive_list_vali,negative_list_vali,x,x = train_test_triplet(kingdom,phylum,class_,order,family,genus,file_name,vali_index,False)
    print('vali: ',len(anchor_list_vali))
    


    data = get_data(file_name)
    anchor_data_train = list2data(anchor_list_train,data,k)
    positive_data_train = list2data(positive_list_train, data, k)
    negative_data_train = list2data(negative_list_train, data, k)

    anchor_data_vali = list2data(anchor_list_vali,data,k)
    positive_data_vali = list2data(positive_list_vali, data, k)
    negative_data_vali = list2data(negative_list_vali, data, k)


    model.fit([anchor_data_train,positive_data_train,negative_data_train],  np.zeros(len(anchor_data_train)), shuffle = False,
      epochs=1 ,batch_size=5000,verbose =1,
      validation_data =([anchor_data_vali,positive_data_vali,negative_data_vali],np.zeros(len(anchor_data_vali))),
   #   callbacks=[callbackmodel(model)],
      sample_weight = np.array(last_weight)
            )


    
if __name__ == "__main__":
    train_data = pd.read_csv('data.csv')
  
    
    k = 6
    kingdom = list(train_data['kingdom'])
    phylum = list(train_data['phylum'])
    class_ = list(train_data['class'])
    order = list(train_data['order'])
    family = list(train_data['family'])
    genus = list(train_data['genus'])
    all_species = list(train_data.iloc[:, -1])    
    file_name = ['_'.join(['_'.join(_.split())]) for _ in all_species]
 
    l = list(range(len(file_name)))
    AT = list(pd.read_table('k'+str(k)+'.txt', header=None, index_col=0).T)
     
    train(kingdom,phylum,class_,order,family,genus,file_name,l,k)

 

     

