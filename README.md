# SAINT
**SAINTis a weakly-supervised learning method where the embedding function is learned automatically from the easily-acquired data.Compared to existing deep learning-based alignment-free method, SAINT doesn’t require tedious labors to collect accurate alignment distances to train.SAINT is more computationally fast and memory efficient because
sequence data are operated in a compressed embedding space which is much faster to retrieval and succinct to store.**
## Version Release Notes
- Version 1.0
1. This is the first version of SAINT pipeline. 
2. This An demo of SAINT running is given here. 
## Package installation and configuration
- Pre-install running environment
1.Unix or Linux operating system.
2.Python 2.7 or above.
- Detailed steps
1.Download the source code to your directory, e.g: ’/home/user/SAINT’.
2.Enter your specified directory: $ cd /home/user/SAINT
3.Extract the tar file: $ ungiz resource/kmer.zip
4.Enter the directory: $ cd /home/user/SAINT/code
5.If your operating system has multiple Python version, please be sure your Python version at least 2.7 or above.
## The demo of SAINT
The dataset was download from NCBI. For the 200 bacteria genomes, Saint uses KMC tool to convert fasta file into kmer frequency file.
**Run SAINT**
1.Run SAINT to get model.
Create a new folder to put output file
$ mkdir ./model
Run triplet_model.py
$ python triplet_model.py
