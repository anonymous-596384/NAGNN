# NAGNN 
Source code for NAGNN (under review)

FileList:

	NAGNN		:	The source code of NAGNN model
	datasets	:	Datasets we used in NAGNN.
	readMe		:	This file.

The project consists of two folders: NAGNN and datasets. All the source codes are in folder NAGNN, and the datasets are in folder datasets.
Besides, the source code is based on GCN model.

====
Environment
====
python-3.6.5

tensorflow-1.13.0

====
NAGNN source code
====

FileList in NAGNN source-code folder:

	paramsConfigPython	:	parameters setting file, you can modify the parameters in this file
	mainD.py		:	the entry of pretraining for discriminator D
	D_pre_Training.py	:	the pretraining body for discriminator D
	mainG.py		:	the entry of pretraining for generator G
	G_pre_Training.py	:	the pretraining body for generator G
	mainGAN.py		:	the entry of training NAGNN
	AdGCNTraining.py	:	the training body for NAGNN
	generator.py		:	the generator model, in which we define the generator G
	discriminator.py	:	the discriminator model, in which we define the discriminator D
	processTools.py		:	some tool functions for datasets processing
  

The training steps:

1. Pretraining for discriminator D : we first train an initialization for discriminator D, that is to pretrain the GCN model in D on the training dataset. The main entry is mainD.py.

2. Pretraining for generator G : we train an initialization for generator G, that is to pretrain MLPs in G by predicting the raw features. The main entry is mainG.py.

3. Training NAGNN : finally we train the NAGNN model based on the pretraining of D and G, and then to classify nodes for test. The main entry is mainGAN.py. Finally, the model would predict the classfication for test nodes.

