#encoding=utf-8
'''

'''
import numpy as np
import os 
import configparser
import G_pre_Training


cf = configparser.SafeConfigParser()
cf.read("paramsConfigPython")

root_dir = cf.get("param", "root_dir") 
dataset = cf.get("param", "dataset") # dataset name
root_dir = root_dir + dataset + '/'
gpu = cf.get("param", "gpu") # gpu id

os.environ["CUDA_VISIBLE_DEVICES"] = gpu


hid_units = [int(i) for i in cf.get("param", "hid_units").split(',')] 
randomVectorsLen = cf.getint("param", "randomVectorsLen") # the length for fake sampled vector
covSampling = cf.getfloat("param", "covSampling") #the default covariance for normal distribution for sampling, default is 1.0
lr_pre_D = cf.getfloat("param", "lr_pre_D") # learning rate
lr_pre_G = cf.getfloat("param", "lr_pre_G") # learning rate
lr_D = cf.getfloat("param", "lr_D") # learning rate
lr_G = cf.getfloat("param", "lr_G") # learning rate
un_weight = cf.getfloat("param", "un_weight") # the weight for unsupervised loss
fake_weight = cf.getfloat("param", "fake_weight") # the weight for fake data
fake_times = cf.getint("param", "fake_times") # the time for fake nodes wrt one real node
pre_l2_coef = cf.getfloat("param", "pre_l2_coef") # coefficient of l2 regularization 
GAN_D_l2_coef = cf.getfloat("param", "GAN_D_l2_coef") # coefficient of l2 regularization 
GAN_G_l2_coef = cf.getfloat("param", "GAN_G_l2_coef") # coefficient of l2 regularization 
l2_coef_param = cf.getfloat("param", "l2_coef_param") # coefficient of l2 regularization 
dropout = cf.getfloat("param", "dropout") # dropout
D_pretrain_epoch = cf.getint("param", "D_pretrain_epoch") # pretrain epoch for D
D_pretrain_patience = cf.getint("param", "D_pretrain_patience") # pretrain patience epoch for D
G_pretrain_epoch = cf.getint("param", "G_pretrain_epoch") # pretrain epoch for G
G_pretrain_param = cf.getfloat("param", "G_pretrain_param") # G_pretrain_param
G_pretrain_min_loss  = cf.getfloat("param", "G_pretrain_min_loss") # G_pretrain_param
epoch_num = cf.getint("param", "epoch_num") # epoch
inner_epoch_D = cf.getint("param", "inner_epoch_D") # in each batch data, the inner training epochs for D
inner_epoch_D_su = cf.getint("param", "inner_epoch_D_su") #
inner_epoch_G = cf.getint("param", "inner_epoch_G") # in each batch data, the inner training epochs for G
patience = cf.getint("param", "patience") # patience for training 
batch_size = cf.getint("param", "batch_size") # batch size for data (mainly for unlabeled data)
shuffleForBatch = cf.getboolean("param", "shuffleForBatch") # is shuffle for data (mainly for unlabeled data)
discountCE = cf.getfloat("param", "discountCE") # discount for cross entropy

randomWalkTimes = cf.getint("param", "randomWalkTimes") # discount for cross entropy
randomWalkHop = cf.getint("param", "randomWalkHop") # discount for cross entropy
randomWalkWeightParam = cf.getfloat("param", "randomWalkWeightParam") # discount for cross entropy

# pre-train for G
G_pre_Training.adGCNTraining(
    root_dir, 
    dataset, 
    hid_units, 
    randomVectorsLen, 
    covSampling,
    lr_pre_D,
    lr_pre_G,
    lr_D,
    lr_G, 
    un_weight,
    fake_weight,
    fake_times,
    pre_l2_coef, 
    GAN_D_l2_coef,
    GAN_G_l2_coef,
    l2_coef_param,
    dropout,
    D_pretrain_epoch, 
    D_pretrain_patience,
    G_pretrain_epoch, 
    G_pretrain_param,
    G_pretrain_min_loss,
    epoch_num, 
    inner_epoch_D,
    inner_epoch_D_su,
    inner_epoch_G, 
    patience, 
    batch_size, 
    shuffleForBatch,
    discountCE,
    randomWalkTimes,
    randomWalkHop,
    randomWalkWeightParam)

