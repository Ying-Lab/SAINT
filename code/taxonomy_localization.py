# -*- coding: utf-8 -*-

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
         m =pd.read_csv('kmer/'+file_name[i]+'_k6.txt', header=None,sep='\t',index_col=[0]).T
         new_data = (m.reindex(columns=AT, fill_value=0)).iloc[0]
     #   data[file_name[i]] =np.around((new_data)*10000,6)
         data[file_name[i]] =np.around((new_data/new_data.sum())*10000,6)
    return data


if __name__ == "__main__":

    train_data = pd.read_csv('data.csv')
    all_species = list(train_data.iloc[:, -1])
    all_name_ = ['_'.join(['_'.join(_.split())]) for _ in all_species]

    test_name = list(pd.read_table('test_name.txt',header = None,index_col=0).T)
    train_name = list(set(all_name_).difference(set(test_name)))
    phylum_ = list(train_data['phylum'])
    class_ = list(train_data['class'])
    order_ = list(train_data['order'])
    family_ = list(train_data['family'])
    genus_ = list(train_data['genus'])
    AT = list(pd.read_table('k'+str(6)+'.txt', header=None, index_col=0).T)
    data = get_data(all_name_)

    test_category = {}
    for test_iterm in test_name:
        test_category[test_iterm] = genus_[all_name_.index(test_iterm)]

    genus = []
    all_name = []
    cc = list(zip(genus_,all_name_))
    mid = []
    for test_iterm in train_name:
        mid.append(cc[all_name_.index(test_iterm)])
    genus[:],all_name[:]= zip(*mid)   
    non_unique_train = []
    for dup in sorted(list_duplicates(genus)):
        non_unique_train.append(dup)
    
    model1 = load_model('model/best_model.h5',custom_objects={'identity_loss':identity_loss,'eulidean_distance':eulidean_distance})
    model = Model(inputs=model1.input[0],outputs=model1.get_layer('Dense_9').get_output_at(0))
    g = 0
    f = 0
    o = 0
    c =0
    p = 0
    n = 0
    file = open('output/predict_taxonomy.txt','w')
    file.write('predict_phylum'+'\t'+'predict_class'+'\t'+'predict_order'+'\t'+'predict_family'+'\t'+'predict_genus'+'\t'+'species'+'\n')
    for anchor_spiece in test_name:
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
        file.write(phylum_[genus_.index(all_[0][0])]+'\t'+class_[genus_.index(all_[0][0])]+'\t'+order_[genus_.index(all_[0][0])]+'\t'+family_[genus_.index(all_[0][0])]+'\t'+all_[0][0]+'\t'+anchor_spiece+'\n')
    file.close()
    file2 = open('output/Accuracy.txt','w')
    n = len(test_name)    
    file2.write('SAINT taxonomy localization result:'+'\n')  
    file2.write('Layer'+'\t'+'phylum'+'\t'+'class'+'\t'+'order'+'\t'+'family'+'\t'+'genus'+'\n')
    file2.write('True_Number/all_number'+'\t'+str(o)+'/'+str(n)+'\t'+str(f)+'/'+str(n)+'\t'+str(g)+'/'+str(n)+'\t' +str(c)+'/'+str(n)+'\t'+str(p)+'/'+str(n)+'\n')
    file2.write('Accuracy'+'\t'+str(g/n)+'\t'+str(f/n)+'\t'+str(o/n)+'\t'+str(c/n)+'\t'+str(p/n))
    file2.close()


              
                    
    
