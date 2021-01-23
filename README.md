# SAINT

**SAINT is a weakly-supervised learning method where the embedding function is learned automatically from the easily-acquired data.Compared to existing deep learning-based alignment-free method, SAINT doesn’t require tedious labors to collect accurate alignment distances to train.SAINT is more computationally fast and memory efficient because
sequence data are operated in a compressed embedding space which is much faster to retrieval and succinct to store.**

**Compared to existing alignment-free sequence comparison methods,SAINT offers following advantages:**

 1. SAINTis more computationally fast and memory efficient because sequence data are operated in a compressed embedding space which is much faster to retrieval and succinct to store. 

 2. SAINTis a weakly-supervised learning method where the embedding function is learned automatically from the easily-acquired data. Compared to existing deep learning-based alignment-free method, SAINT doesn’t require tedious labors to collect accurate alignment distances to train. 

## Version Release Notes

- Version 1.0

 1. This is the first version of SAINT pipeline. 

 2. An demo of SAINT running is given here. 

## Package installation and configuration

- Pre-install running environment

 1. Unix or Linux operating system.
 
 2. CPU is enough for calculation.

 3. Python 3 or above.
 
 4. Packages like sys, optparse, os, random, numpy, pandas, collections, keras and sklearn need to be prepared.

- Detailed steps

 1. Download the source code to your directory, e.g: ’/home/user/SAINT’.

 2. Enter your specified directory: 

    >```   
    >   $ cd /home/user/SAINT 
    >```  

 3. Extract the zip file: 

    >```   
    >   $ unzip ./resource/kmer.zip
    >```  

 4. If your operating system has multiple Python version, please be sure your Python version at least 3 or above.

## The demo of SAINT

The dataset was download from NCBI. For the 232 bacteria genomes, Saint uses KMC tool to convert fasta file into kmer frequency file [here](https://github.com/Ying-Lab/SAINT/tree/main/resource/kmer.zip).

**Run SAINT**

1. Usage of SAINT

- The main running command are triplet_model.py and taxonomy_localization.py with following options:

     -h, --help: show this help message and exit
     
     -i, --inputcsv: the taxomony of the input data
     
     -d, --kmer_frequency_dir: the dir of kmer frequency.
     
     -t, --test_name: the list of test name.
     
     -k, --kofKTuple: the value k of KTuple
     
     -e, --epochNum: the number of epoch.
     
     -o, --output: output dir.

2. Run SAINT to get model.

    Create a new folder to put model file

    >```   
    >   $ mkdir output
    >```  

    Run triplet_model.py
    >```  
    >   $ python code/triplet_model.py -i resource/data.csv -d resource/kmer/ -t resource/test_name.txt -k 6 -e 30 -o output/

    >```  
 

3. Predict taxonomy of unknown species and Calculate the performance of SAINT results.`  

    Run taxonomy_localization.py
    
    >```   
    >   $ python code/taxonomy_localization.py -i resource/data.csv -d resource/kmer/ -t resource/test_name.txt  -o output/

    >``` 

    The output are ./output/predict_taxonomy.txt and ./output/Accuracy.txt.

