#!/usr/bin/env python
# coding: utf-8


def Clustered_model(NUM_CLIENTS, NUM_CLUSTERS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic,  BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_clusters, client_train_dataset, Clustered_file_path):
    
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
    from worker_functions import model_prediction_cluster


    ####### Initalize the results dataframe
    Clustered_results_df = pd.DataFrame({
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


    
    
    

    
    ###################################################################################### Start the learning ######################################################################################
    start_time = time.time()

    accuracy_dic = {}
    loss_dic = {}

    rand_dic = {}
    avg_dic = {}

    communication_cost = {}

    time_calc_dic = {}

    time_calc_dic['Param_split'] = []
    time_calc_dic['Subtotal_calc'] = []
    time_calc_dic['Avg_calc'] = []
    time_calc_dic['Model_train'] = []
    time_calc_dic['Final_sync'] = []


    for index in range(NUM_CLIENTS):
        model_name = "model_" + str(index)
        communication_cost[model_name] = 0






    ######################################## Init model ########################################
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





    ### Change the manual entering
    layer1_input_size = len(prev_dic['model_0'][0][0])
    layer1_output_size = len(prev_dic['model_0'][0][0][1])
    layer1_bias_size = len(prev_dic['model_0'][0][1])
    layer2_input_size = len(prev_dic['model_0'][1][0])
    layer2_output_size = len(prev_dic['model_0'][1][0][1])
    layer2_bias_size = len(prev_dic['model_0'][1][1])
    layer3_input_size = len(prev_dic['model_0'][2][0])
    layer3_output_size = len(prev_dic['model_0'][2][0][1])
    layer3_bias_size = len(prev_dic['model_0'][2][1])

    dim_list = [[[layer1_input_size, layer1_output_size], layer1_bias_size], [[layer2_input_size, layer2_output_size], layer2_bias_size], [[layer3_input_size, layer3_output_size], layer3_bias_size] ]




    ####################################################################################### Iterative training #######################################################################################
    for round_num in range(NUM_ROUNDS):

        print(f"The round: {round_num + 1}")

        accuracy_dic[round_num] = {}
        loss_dic[round_num] = {}

        for cluster in range(NUM_CLUSTERS):
            cluster_len = len(client_clusters[cluster])
            rand_dic[cluster] = np.random.randint(100, size=(cluster_len, cluster_len))
            avg_dic[cluster] = rand_dic[cluster]/ rand_dic[cluster].sum(axis=1)[:,None] 



        ######################################## Split the parameters according to random ratios ########################################
        sp_time = time.time()
        split_dict = {}

        for cluster in range(NUM_CLUSTERS):

            for index in range(len(client_clusters[cluster])):
                model_name = "model_" + str(client_clusters[cluster][index])
                client_name = "client_" + str(client_clusters[cluster][index])
                split_dict[model_name] = []


                for layer in range(len(dim_list)):
                    layer_list = []
                    for itr1 in range(2):
                        if itr1 == 0:
                            split_arr = np.multiply.outer(prev_dic[model_name][layer][itr1],avg_dic[cluster][index,:])

                        else:
                            split_arr = np.multiply.outer(prev_dic[model_name][layer][itr1],avg_dic[cluster][index,:])

                        layer_list.append(split_arr)

                    split_dict[model_name].append(layer_list) 
                split_dict[model_name] = np.array(split_dict[model_name], dtype=object)

        sp_elap = time.time() - sp_time
        time_calc_dic['Param_split'].append(sp_elap)
        





        ######################################## Calculating partital sums ########################################      
        calc_time = time.time()

        calc_dic = {}

        for cluster in range(NUM_CLUSTERS):

            cluster_len = len(client_clusters[cluster])
            for index in range(cluster_len):
                model_name = "model_" + str(client_clusters[cluster][index])
                client_name = "client_" + str(client_clusters[cluster][index])

                comm_count = 0                
                calc_dic[model_name] = [[] for layer in range(len(dim_list))]


                for layer in range(len(dim_list)):
                    for itr1 in range(2):
                        if itr1 == 0:
                            sum_calc = np.zeros([dim_list[layer][itr1][0], dim_list[layer][itr1][1]], dtype= float)
                            for user in range(cluster_len):
                                temp_name = "model_" + str(client_clusters[cluster][user])
                                sum_calc = sum_calc + split_dict[temp_name][layer][itr1][:,:,index]

                            calc_dic[model_name][layer].append(sum_calc)
                            comm_count += dim_list[layer][itr1][0] * dim_list[layer][itr1][1] * (cluster_len -1)

                        else:
                            sum_calc = np.zeros([dim_list[layer][itr1]], dtype= float)
                            for user in range(cluster_len):
                                temp_name = "model_" + str(client_clusters[cluster][user])
                                sum_calc = sum_calc + split_dict[temp_name][layer][itr1][:,index]

                            calc_dic[model_name][layer].append(sum_calc)
                            comm_count += dim_list[layer][itr1] * (cluster_len -1)

                communication_cost[model_name] = communication_cost[model_name] + comm_count
                calc_dic[model_name] = np.array(calc_dic[model_name], dtype=object)


        calc_elap = time.time() - calc_time
        time_calc_dic['Subtotal_calc'].append(calc_elap)





        ######################################## Calculating the parameter averages ########################################  
        fin_time = time.time()  

        final_dic = {}

        for cluster in range(NUM_CLUSTERS):

            cluster_size = len(client_clusters[cluster])

            for index in range(cluster_size):
                model_name = "model_" + str(client_clusters[cluster][index])
                client_name = "client_" + str(client_clusters[cluster][index])

                final_dic[model_name] = [[] for layer in range(len(dim_list))]
                comm_count = 0



                for layer in range(len(dim_list)):
                    for itr1 in range(2):
                        if itr1 == 0:
                            sum_calc = np.zeros([dim_list[layer][itr1][0], dim_list[layer][itr1][1]], dtype= float)

                            for user in range(cluster_size):
                                temp_name = "model_" + str(client_clusters[cluster][user])
                                sum_calc = sum_calc + calc_dic[temp_name][layer][itr1]

                            avg = sum_calc/cluster_size

                            final_dic[model_name][layer].append(avg)
                            prev_dic[model_name][layer][itr1] = avg
                            comm_count += dim_list[layer][itr1][0] * dim_list[layer][itr1][1] * (cluster_size -1)

                        else:
                            sum_calc = np.zeros([dim_list[layer][itr1]], dtype= float)

                            for user in range(cluster_size):
                                temp_name = "model_" + str(client_clusters[cluster][user])
                                sum_calc = sum_calc + calc_dic[temp_name][layer][itr1]

                            avg = sum_calc/cluster_size

                            final_dic[model_name][layer].append(avg)
                            prev_dic[model_name][layer][itr1] = avg
                            comm_count += dim_list[layer][itr1] * (cluster_size -1)


                communication_cost[model_name] = communication_cost[model_name] + comm_count
                final_dic[model_name] = np.array(final_dic[model_name], dtype=object)


        fin_elap = time.time() - fin_time
        time_calc_dic['Avg_calc'].append(fin_elap)





        ######################################## New model with parameter averages ########################################                        
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


    
    
    

    ####################################################################################### Final syncing #######################################################################################
    ad_time  = time.time()
    
    ### To make the values of final_dic and prev_dic the same  
    prev_dic = copy.deepcopy(final_dic)

    ad_elap = time.time() - ad_time        
    time_calc_dic['Final_sync'].append(ad_elap) 


    

    #### Time calculation ####
    elapsed_time = time.time() - start_time 
    training_time = str(timedelta(seconds=elapsed_time))


    sp_elap_tot = sum(time_calc_dic['Param_split'])
    calc_elap_tot = sum(time_calc_dic['Subtotal_calc'])
    fin_elap_tot = sum(time_calc_dic['Avg_calc'])
    nm_elap_tot = sum(time_calc_dic['Model_train'])
    ad_elap_tot = sum(time_calc_dic['Final_sync'])

    print(f'Initial model training time - {str(timedelta(seconds= init_elap_time))}')
    print(f'Parameter splitting time - {str(timedelta(seconds=sp_elap_tot))}')
    print(f'Sub total calculation time - {str(timedelta(seconds=calc_elap_tot))}')
    print(f'Average calculation time - {str(timedelta(seconds=fin_elap_tot))}')
    print(f'Model training time - {str(timedelta(seconds=nm_elap_tot))}')
    print(f'Final syncing time - {str(timedelta(seconds=ad_elap_tot))}')


    

    
    
    ####################################################################################### Calculating the results #######################################################################################
    with Manager() as manager:
        multi_precision_dic = manager.dict()    
        multi_recall_dic = manager.dict() 
        multi_f1_score_dic = manager.dict() 
        multi_test_acc_dic = manager.dict() 

        pool = Pool() 


        pool.starmap(model_prediction_cluster, zip(range(NUM_CLUSTERS), repeat(client_clusters), repeat(multi_precision_dic), repeat(multi_recall_dic), repeat(multi_f1_score_dic), repeat(multi_test_acc_dic), repeat(prev_dic), repeat(X_test), repeat(y_test)))
        pool.close()
        pool.join()


        precision_dic = copy.deepcopy(multi_precision_dic)
        recall_dic = copy.deepcopy(multi_recall_dic)
        f1_score_dic = copy.deepcopy(multi_f1_score_dic)
        test_acc_dic = copy.deepcopy(multi_test_acc_dic)


    
    
    

    ####################################################################################### Saving the results #######################################################################################
    Clustered_results_df.loc[len(Clustered_results_df.index)] = [no_of_training_samples, number_of_test_data, NUM_CLIENTS, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, precision_dic, recall_dic, f1_score_dic, test_acc_dic, elapsed_time, communication_cost]


    try:
        with open(Clustered_file_path, "rb+") as pickle_file:
            clustered_results = pickle.load(pickle_file)
            clustered_results.loc[len(clustered_results.index)] = [no_of_training_samples, number_of_test_data, NUM_CLIENTS, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, precision_dic, recall_dic, f1_score_dic, test_acc_dic, elapsed_time, communication_cost]
            pickle_file.seek(0)
            pickle.dump(clustered_results, pickle_file)

    except FileNotFoundError as fnfe:
        with open(Clustered_file_path, "wb") as pickle_file:
            pickle.dump(Clustered_results_df, pickle_file)

    pickle_file.close()