#encoding=utf-8
'''
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 
import numpy as np 
import os 
import discriminator
import generator
import processTools
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def variablesInit(options):
    """
    init variables
    """
    variablesMap={}
    D_W0 = tf.Variable(tf.random_uniform([options['feature_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W0") # shape=(nNodes,dim)
    D_W1 = tf.Variable(tf.random_uniform([options['hid_units'][0], options['all_class_num']], -0.01, 0.01), dtype=tf.float32, name="D_W1") # shape=(nNodes,dim)
    variablesMap["D_W0"]=D_W0
    variablesMap["D_W1"]=D_W1
    
    G_MLP_W = tf.Variable(tf.random_uniform([options['feature_num'], options['feature_num']], -0.01, 0.01), dtype=tf.float32, name="G_MLP_W") # shape=(concat_len,feature_num)
    G_MLP_b = tf.Variable(tf.random_uniform([options['feature_num']], -0.01, 0.01), dtype=tf.float32, name="G_MLP_b") # shape=(feature_num)
    variablesMap["G_MLP_W"]=G_MLP_W
    variablesMap["G_MLP_b"]=G_MLP_b
    
    theta_D = [D_W0, D_W1]
    theta_G = [G_MLP_W, G_MLP_b]
    
    return variablesMap, theta_D, theta_G


def adGCNTraining(
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
        randomWalkWeightParam,
        activation=tf.nn.elu):
    """
    training
    """
    options = locals().copy() 
    cur_path = os.getcwd()
    pre_train_checkpt_file=cur_path + "/modelsSave/pre_model_save_"+dataset+".ckpt"  
    checkpt_file=cur_path + "/modelsSave/model_save_"+dataset+".ckpt"  
    
    if dataset=='bib':
        nodeFile = root_dir + 'graph.node'
        edgeFile = root_dir + 'graph.edge'
        trainFile = root_dir + 'train_nodes'
        valFile = root_dir + 'val_nodes'
        testFile = root_dir + 'test_nodes'
        adj, features, features_01, y_train, y_val, y_test, train_mask, val_mask, test_mask = processTools.load_data_dblp(nodeFile, edgeFile, trainFile, valFile, testFile)
    else:
        adj, features_ori, y_train, y_val, y_test, train_mask, val_mask, unlabeled_mask, test_mask, idx_unlabeled = processTools.load_data_unlabeled(root_dir, dataset)
        features, spars, features_01 = processTools.preprocess_features(features_ori) 
        adj = adj.toarray() 
    
    node_num = features.shape[0] 
    feature_num = features.shape[1] 
    class_num = y_train.shape[1] 
    all_class_num = class_num+1 
    options['node_num'] = node_num
    options['feature_num'] = feature_num
    options['class_num'] = class_num
    options['all_class_num'] = all_class_num
    
    options['GAN_l2'] = l2_coef_param / class_num
    
    # placeholders
    ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
    G_pretrain_weight = tf.placeholder(dtype=tf.float32, shape=())
    isPreTrain_flag = tf.placeholder(dtype=tf.float32, shape=())
    
    idealFeatures_su = tf.placeholder(dtype=tf.float32, shape=(None, None)) 
    lbl_in_su = tf.placeholder(dtype=tf.int32, shape=(None, options['all_class_num'])) 
    lbl_1_in_su = tf.placeholder(dtype=tf.int32, shape=(None, options['all_class_num'])) 
    adj_0_su = tf.placeholder(dtype=tf.int32, shape=(None, None)) 
    mask_0_su = tf.placeholder(dtype=tf.float32, shape=(None, None)) 
    adj_1_su = tf.placeholder(dtype=tf.int32, shape=(None, None)) 
    mask_1_su = tf.placeholder(dtype=tf.float32, shape=(None, None))
    features_array_su = tf.placeholder(dtype=tf.float32, shape=(None, options['feature_num'])) 
    feature_self_mask_su = tf.placeholder(dtype=tf.float32, shape=(None,)) 
    feature_cosp_mask_su = tf.placeholder(dtype=tf.float32, shape=(None, None)) 
    
    
    variablesMap, theta_D, theta_G = variablesInit(options)
    
    # build model
    fakeFeaturesArray_su, G_loss_pretrain_su, G_lossL2_pretrain_su = generator.generatorFeatureByAdjModel(options, variablesMap, idealFeatures_su, features_array_su, feature_self_mask_su, feature_cosp_mask_su, covSampling, G_pretrain_weight)
    G_loss_pretrain = G_lossL2_pretrain_su
    
    D_loss_su_r, D_lossL2_su_r, D_accuracy_su_r, D_predLabels_su_r, D_trueLabels_su_r = discriminator.discriminatorModel(options, variablesMap, ffd_drop, lbl_in_su, lbl_1_in_su, adj_0_su, mask_0_su, adj_1_su, mask_1_su, features_array_su, isSupervised_flag=1.0, isReal_flag=1.0, isOnlyAddZerosCol_flag=1.0, isDLoss_flag=1.0, isPreTrain_flag=isPreTrain_flag)
    D_lossL2_su_f = 0.0
    for i in range(fake_times): 
        # Generator
        fakeFeaturesArray_su_i, G_loss_pretrain_su_i, G_lossL2_pretrain_su_i = generator.generatorFeatureByAdjModel(options, variablesMap, idealFeatures_su, features_array_su, feature_self_mask_su, feature_cosp_mask_su, covSampling, G_pretrain_weight)
        # Discriminator
        D_loss_su_f, D_lossL2_su_f_i, D_accuracy_su_f, D_predLabels_su_f, D_trueLabels_su_f = discriminator.discriminatorModel(options, variablesMap, ffd_drop, lbl_in_su, lbl_1_in_su, adj_0_su, mask_0_su, adj_1_su, mask_1_su, fakeFeaturesArray_su_i, isSupervised_flag=1.0, isReal_flag=0.0, isOnlyAddZerosCol_flag=0.0, isDLoss_flag=1.0, isPreTrain_flag=isPreTrain_flag)
        D_lossL2_su_f += D_lossL2_su_f_i
    D_lossL2_su_f = D_lossL2_su_f / fake_times
    D_loss = D_lossL2_su_r + fake_weight * D_lossL2_su_f
    
    G_loss_su, G_lossL2_su, G_accuracy_su, _2, _3 = discriminator.discriminatorModel(options, variablesMap, ffd_drop, lbl_in_su, lbl_1_in_su, adj_0_su, mask_0_su, adj_1_su, mask_1_su, fakeFeaturesArray_su, isSupervised_flag=1.0, isReal_flag=1.0, isOnlyAddZerosCol_flag=1.0, isDLoss_flag=0.0, isPreTrain_flag=isPreTrain_flag) 
    G_loss = G_lossL2_su
    
    # pretrain
    G_pretrainer_L2 = tf.train.AdamOptimizer(learning_rate=options['lr_pre_G']).minimize(G_loss_pretrain, var_list=theta_G)
    G_pretrainer = tf.train.AdamOptimizer(learning_rate=options['lr_pre_G']).minimize(G_loss_pretrain_su, var_list=theta_G)
    D_pretrainer = tf.train.AdamOptimizer(learning_rate=options['lr_pre_D']).minimize(D_lossL2_su_r, var_list=theta_D)
    # train
    G_trainer = tf.train.AdamOptimizer(learning_rate=options['lr_G']).minimize(G_loss, var_list=theta_G)
    D_trainer = tf.train.AdamOptimizer(learning_rate=options['lr_D']).minimize(D_loss, var_list=theta_D)
    
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0
    vacc_early_model = 0.0
    vlss_early_model = 0.0
    d_loss_early_model = 0.0
    g_loss_early_model = 0.0
    
    # pre-train D
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        
        start_time=time.time()
        print('Start time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        labels_combine = y_train + y_val + y_test 
        labels_combine_sum = np.sum(labels_combine, axis=0) # shape=(7,)
        max_index = np.argmax(labels_combine_sum) 
        mi_f1_labels = [i for i in range(class_num) if i!=max_index] 
        
        adjSelf, adjSelfNor = processTools.preprocess_adj_normalization(adj)
        neis, neis_mask, neis_mask_nor, neighboursDict = processTools.processNodeInfo(adjSelf, adjSelfNor, node_num)
        
        idealFeatures_tr, idealFeatures_01_tr, lbl_tr, lbl_1_tr, adj_0_tr, mask_0_tr, mask_0_nor_tr, adj_1_tr, mask_1_tr, mask_1_nor_tr, features_array_tr, feature_self_mask_tr, feature_cosp_mask_tr = processTools.generateDataByGivenNodes_su_featuresProcess(adjSelf, adjSelfNor, train_mask, features, features_01, node_num, neis, neis_mask, neis_mask_nor, y_train)
        idealFeatures_val, idealFeatures_01_val, lbl_val, lbl_1_val, adj_0_val, mask_0_val, mask_0_nor_val, adj_1_val, mask_1_val, mask_1_nor_val, features_array_val, feature_self_mask_val, feature_cosp_mask_val = processTools.generateDataByGivenNodes_su_featuresProcess(adjSelf, adjSelfNor, val_mask, features, features_01, node_num, neis, neis_mask, neis_mask_nor, y_val)
        idealFeatures_ts, idealFeatures_01_ts, lbl_ts, lbl_1_ts, adj_0_ts, mask_0_ts, mask_0_nor_ts, adj_1_ts, mask_1_ts, mask_1_nor_ts, features_array_ts, feature_self_mask_ts, feature_cosp_mask_ts = processTools.generateDataByGivenNodes_su_featuresProcess(adjSelf, adjSelfNor, test_mask, features, features_01, node_num, neis, neis_mask, neis_mask_nor, y_test)
        
        print('Start to pretrain D model ... time== ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        pretrain_vacc_mx = 0.0 
        pretrain_vlss_mn = 10000.0 
        pretrain_curr_step = 0
        pretrain_stop_epoch = 0
        for epoch in range(D_pretrain_epoch):
            _, lossL2_pretrain_d, acc_pretrain_d = sess.run([D_pretrainer, D_lossL2_su_r, D_accuracy_su_r], feed_dict={
                ffd_drop: options['dropout'], 
                isPreTrain_flag: 1.0,
                lbl_in_su: lbl_tr, 
                lbl_1_in_su: lbl_1_tr, 
                adj_0_su: adj_0_tr, 
                mask_0_su: mask_0_nor_tr, 
                adj_1_su: adj_1_tr, 
                mask_1_su: mask_1_nor_tr, 
                features_array_su: features_array_tr
                })
            
            pretrain_val_loss, pretrain_val_acc, pretrain_val_predLabels, pretrain_val_trueLabels = sess.run([D_lossL2_su_r, D_accuracy_su_r, D_predLabels_su_r, D_trueLabels_su_r], feed_dict={
                ffd_drop: 0.0,
                isPreTrain_flag: 1.0,
                lbl_in_su: lbl_val,
                lbl_1_in_su: lbl_1_val,
                adj_0_su: adj_0_val,
                mask_0_su: mask_0_nor_val,
                adj_1_su: adj_1_val,
                mask_1_su: mask_1_nor_val,
                features_array_su: features_array_val
                })
            val_micro_f1, val_macro_f1 = processTools.micro_macro_f1_removeMiLabels(pretrain_val_trueLabels, pretrain_val_predLabels, mi_f1_labels)
            
            test_loss, test_acc, test_predLabels, test_trueLabels = sess.run([D_lossL2_su_r, D_accuracy_su_r, D_predLabels_su_r, D_trueLabels_su_r], feed_dict={
                ffd_drop: 0.0,
                isPreTrain_flag: 1.0,
                lbl_in_su: lbl_ts,
                lbl_1_in_su: lbl_1_ts,
                adj_0_su: adj_0_ts,
                mask_0_su: mask_0_nor_ts,
                adj_1_su: adj_1_ts,
                mask_1_su: mask_1_nor_ts,
                features_array_su: features_array_ts
                })
            test_micro_f1, test_macro_f1 = processTools.micro_macro_f1_removeMiLabels(test_trueLabels, test_predLabels, mi_f1_labels)
            
            print('Epoch: %d | D-Pretraining: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f | Test: loss = %.5f, acc = %.5f' %
                (epoch, lossL2_pretrain_d, acc_pretrain_d, pretrain_val_loss, pretrain_val_acc, test_loss, test_acc))
            # 进行val结果的记录
            if pretrain_val_loss<=pretrain_vlss_mn or pretrain_val_acc>=pretrain_vacc_mx:
                if pretrain_val_loss<=pretrain_vlss_mn and pretrain_val_acc>=pretrain_vacc_mx: 
                    print('save-------------------------------------------------------------')
                    saver.save(sess, pre_train_checkpt_file) 
                    pretrain_stop_epoch = epoch
                pretrain_vlss_mn = np.min((pretrain_val_loss, pretrain_vlss_mn))
                pretrain_vacc_mx = np.max((pretrain_val_acc, pretrain_vacc_mx))
                pretrain_curr_step = 0
            else: 
                pretrain_curr_step += 1
                if pretrain_curr_step == D_pretrain_patience: 
                    print('Early stop Pretrain Discriminator! Min loss: ', pretrain_vlss_mn, ', Max accuracy: ', pretrain_vacc_mx)
                    break
        saver.restore(sess, pre_train_checkpt_file) 
        test_loss, test_acc, test_predLabels, test_trueLabels = sess.run([D_lossL2_su_r, D_accuracy_su_r, D_predLabels_su_r, D_trueLabels_su_r], feed_dict={
                ffd_drop: 0.0,
                isPreTrain_flag: 1.0,
                lbl_in_su: lbl_ts,
                lbl_1_in_su: lbl_1_ts,
                adj_0_su: adj_0_ts,
                mask_0_su: mask_0_nor_ts,
                adj_1_su: adj_1_ts,
                mask_1_su: mask_1_nor_ts,
                features_array_su: features_array_ts
                })
        test_micro_f1, test_macro_f1 = processTools.micro_macro_f1_removeMiLabels(test_trueLabels, test_predLabels, mi_f1_labels)
        print('Pretrain early stop epoch == ', pretrain_stop_epoch)
        print('End pretrian Discriminator, Test acc == ', test_acc)
        print('End pretrian Discriminator, ', 'mi-f1 == ', test_micro_f1)
        print('End pretrian Discriminator, ', 'ma-f1 == ', test_macro_f1)
        print('----------------------------------------------------------------------------------')
        

