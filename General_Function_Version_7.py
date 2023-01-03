#!/usr/bin/env python
# coding: utf-8


def General_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset, General_file_path):
    
    
    ####################################################################################### Importing Libraries #######################################################################################
    import sys
    sys.path.append("/scratch/project_2006431/dinaj_files/bin")


    import os
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    import numpy as np
    import pandas as pd

    import tensorflow as tf
#     import tensorflow_federated as tff
    from tensorflow.keras import layers
    from tensorflow.keras import backend as K

    tf.get_logger().setLevel(logging.FATAL)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    np.set_printoptions(precision=3, suppress=True)
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans

    from IPython.display import display
    import matplotlib.pyplot as plt
    import seaborn as sns

    import csv
    import random 
    import pickle

    import nest_asyncio
    nest_asyncio.apply()

    from typing import List, Tuple
    import random
    import collections

    import time
    from datetime import timedelta

    import math
    import copy

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    from multiprocessing import Pool, Manager
    from functools import partial
    from itertools import repeat
    from multiprocessing import Pool, freeze_support


    ####### Important ######
    np.random.seed(0)    


    ####### Import custom functions #######
    from worker_functions import init_model_train
    from worker_functions import iterative_model_fit
    from worker_functions import model_prediction_general


    

    ####### Initalize results dataframe 
    General_results_df = pd.DataFrame({
        'Traing data': [],
        'Test data': [],
        'Number of Clients': [],
        'Batch size': [],
        'Number of epochs': [],
        'Number of rounds': [],
        'anomaly_percentage': [],
        'cluster_anomaly_percentage': [],
        'random_anomaly_dic': [],
        'Precision' : [],
        'Recall' : [],
        'F1_score': [],
        'Accuracy': [],
        'Training Time': [],
        'Number of communications': [],
    })



    
    ####################################################################################### Start the learning #######################################################################################
    start_time = time.time()

    time_calc_dic = {}
    time_calc_dic['Model_train'] = []


    accuracy_dic = {}
    loss_dic = {}


    ####################################################################################### Init model #######################################################################################
    init_start_time = time.time()
    print(f'Inital training')
    with Manager() as manager:
        multi_prev_dic = manager.dict()
        multi_model_dic = manager.dict()
        pool = Pool()
        pool.starmap(init_model_train, zip(range(NUM_CLIENTS), repeat(multi_prev_dic), repeat(client_train_dataset), repeat(NUM_EPOCHS), repeat(BATCH_SIZE)))
        pool.close()
        pool.join()
        stop_time = time.time()   

        prev_dic = multi_prev_dic.copy()

    init_elap_time = time.time() - init_start_time





    ####################################################################################### Iterative training #######################################################################################
    for round_num in range(NUM_ROUNDS):

        print(f"The round: {round_num + 1}")

        accuracy_dic[round_num] = {}
        loss_dic[round_num] = {}


        nm_time = time.time() 

        with Manager() as manager:
            itr_perv_dic = manager.dict()
            pool = Pool() 

            pool.starmap(iterative_model_fit, zip(range(NUM_CLIENTS), repeat(itr_perv_dic), repeat(client_train_dataset), repeat(NUM_EPOCHS), repeat(BATCH_SIZE), repeat(prev_dic)))
            pool.close()
            pool.join()



            new_prev_dic = itr_perv_dic.copy()
            prev_dic = copy.deepcopy(new_prev_dic)

        nm_elap = time.time() - nm_time
        time_calc_dic['Model_train'].append(nm_elap)    


    print('\n')


    
    elapsed_time = time.time() - start_time 
    # training_time = str(timedelta(seconds=elapsed_time))

    
    #### Time calculation ####
    nm_elap_tot = sum(time_calc_dic['Model_train'])

    print(f'Initial model training time - {str(timedelta(seconds= init_elap_time))}')
    print(f'Model training time - {str(timedelta(seconds=nm_elap_tot))}')


    

    ####################################################################################### Calculating th results #######################################################################################
    with Manager() as manager:
        multi_precision_dic = manager.dict()    
        multi_recall_dic = manager.dict() 
        multi_f1_score_dic = manager.dict() 
        multi_test_acc_dic = manager.dict() 

        pool = Pool() 


        pool.starmap(model_prediction_general, zip(range(NUM_CLIENTS), repeat(multi_precision_dic), repeat(multi_recall_dic), repeat(multi_f1_score_dic), repeat(multi_test_acc_dic), repeat(prev_dic), repeat(X_test), repeat(y_test)))
        pool.close()
        pool.join()


        precision_dic = copy.deepcopy(multi_precision_dic)
        recall_dic = copy.deepcopy(multi_recall_dic)
        f1_score_dic = copy.deepcopy(multi_f1_score_dic)
        test_acc_dic = copy.deepcopy(multi_test_acc_dic)


    

    #### communication costs ####
    communication_cost = {}

    for index in range(NUM_CLIENTS):
        model_name = "model_" + str(index)
        communication_cost[model_name] = 0


    
    ####################################################################################### Saving the results #######################################################################################
    General_results_df.loc[len(General_results_df.index)] = [no_of_training_samples, number_of_test_data, NUM_CLIENTS, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, precision_dic, recall_dic, f1_score_dic, test_acc_dic, elapsed_time, communication_cost]


    try:
        with open(General_file_path, "rb+") as pickle_file:
            general_results = pickle.load(pickle_file)
            general_results.loc[len(general_results.index)] = [no_of_training_samples, number_of_test_data, NUM_CLIENTS, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, precision_dic, recall_dic, f1_score_dic, test_acc_dic, elapsed_time, communication_cost]
            pickle_file.seek(0)
            pickle.dump(general_results, pickle_file)

    except FileNotFoundError as fnfe:
        with open(General_file_path, "wb") as pickle_file:
            pickle.dump(General_results_df, pickle_file)

    pickle_file.close()