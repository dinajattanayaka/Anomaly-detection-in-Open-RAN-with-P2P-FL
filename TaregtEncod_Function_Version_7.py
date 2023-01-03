#!/usr/bin/env python
# coding: utf-8


def TaregtEncod_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset, TargetEnc_file_path):
    
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
    from worker_functions import model_prediction_targetEnc


    ####### Initalize the results dataframe
    TargetEnc_results_df = pd.DataFrame({
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
        
        #### Random number generation
        rand_array = np.random.randint(100, size=(NUM_CLIENTS, NUM_CLIENTS))
        avg_array = rand_array/rand_array.sum(axis=1)[:,None]




        ######################################## Split the parameters according to random ratios ########################################
        sp_time = time.time()

        split_dict = {}

        for index in range(NUM_CLIENTS):
            model_name = "model_" + str(index)

            split_dict[model_name] = []


            for layer in range(len(dim_list)):
                layer_list = []
                for itr1 in range(2):
                    if itr1 == 0:
                        split_arr = np.multiply.outer(prev_dic[model_name][layer][itr1],avg_array[index,:])

                    else:
                        split_arr = np.multiply.outer(prev_dic[model_name][layer][itr1],avg_array[index,:])

                    layer_list.append(split_arr)

                split_dict[model_name].append(layer_list) 
            split_dict[model_name] = np.array(split_dict[model_name], dtype=object)


        sp_elap = time.time() - sp_time
        time_calc_dic['Param_split'].append(sp_elap)




        ######################################## Calculating partital sums ########################################       
        calc_time = time.time() 

        calc_dic = {}

        for index in range(NUM_CLIENTS):
            model_name = "model_" + str(index)

            calc_dic[model_name] = [[] for layer in range(len(dim_list))]

            comm_count = 0


            for layer in range(len(dim_list)):
                for itr1 in range(2):
                    if itr1 == 0:
                        sum_calc = np.zeros([dim_list[layer][itr1][0], dim_list[layer][itr1][1]], dtype= float)
                        for user in range(NUM_CLIENTS):
                            temp_name = "model_" + str(user)
                            sum_calc = sum_calc + split_dict[temp_name][layer][itr1][:,:,index]


                        calc_dic[model_name][layer].append(sum_calc)
                        comm_count += dim_list[layer][itr1][0] * dim_list[layer][itr1][1] * (NUM_CLIENTS -1)

                    else:
                        sum_calc = np.zeros([dim_list[layer][itr1]], dtype= float)
                        for user in range(NUM_CLIENTS):
                            temp_name = "model_" + str(user)
                            sum_calc = sum_calc + split_dict[temp_name][layer][itr1][:,index]


                        calc_dic[model_name][layer].append(sum_calc)
                        comm_count += dim_list[layer][itr1] * (NUM_CLIENTS -1)


            communication_cost[model_name] = communication_cost[model_name] + comm_count
            calc_dic[model_name] = np.array(calc_dic[model_name], dtype=object)   



        calc_elap = time.time() - calc_time
        time_calc_dic['Subtotal_calc'].append(calc_elap)





        ######################################## Calculating the parameter averages ########################################      
        fin_time = time.time() 

        final_dic = {}

        for index in range(NUM_CLIENTS):

            sub_comm_count = 0

            model_name = "model_" + str(index)
            final_dic[model_name] = [[] for layer in range(len(dim_list))]


            for layer in range(len(dim_list)):
                for itr1 in range(2):
                    if itr1 == 0:
                        sum_calc = np.zeros([dim_list[layer][itr1][0], dim_list[layer][itr1][1]], dtype= float)

                        for user in range(NUM_CLIENTS):
                            temp_name = "model_" + str(user)
                            sum_calc = sum_calc + calc_dic[temp_name][layer][itr1]

                        avg = sum_calc/NUM_CLIENTS

                        final_dic[model_name][layer].append(avg)
                        prev_dic[model_name][layer][itr1] = avg
                        sub_comm_count += dim_list[layer][itr1][0] * dim_list[layer][itr1][1] * (NUM_CLIENTS -1)


                    else:
                        sum_calc = np.zeros([dim_list[layer][itr1]], dtype= float)

                        for user in range(NUM_CLIENTS):
                            temp_name = "model_" + str(user)
                            sum_calc = sum_calc + calc_dic[temp_name][layer][itr1]

                        avg = sum_calc/NUM_CLIENTS

                        final_dic[model_name][layer].append(avg)
                        prev_dic[model_name][layer][itr1] = avg
                        sub_comm_count += dim_list[layer][itr1] * (NUM_CLIENTS -1)


            communication_cost[model_name] = communication_cost[model_name] + sub_comm_count
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
        multi_predict_dic = manager.dict()
        pool = Pool() 


        pool.starmap(model_prediction_targetEnc, zip(range(1), repeat(multi_predict_dic), repeat(prev_dic), repeat(X_test)))
        pool.close()
        pool.join()

        prediction_dic = copy.deepcopy(multi_predict_dic)


    y_pred_fl = np.argmax(prediction_dic['model_0'], axis = 1)

    accuracy = accuracy_score(y_test, y_pred_fl)
    recall = recall_score(y_test, y_pred_fl)
    f1 = f1_score(y_test, y_pred_fl)
    precision = precision_score(y_test, y_pred_fl)




    ####################################################################################### Saving the results #######################################################################################
    TargetEnc_results_df.loc[len(TargetEnc_results_df.index)] = [no_of_training_samples, number_of_test_data, NUM_CLIENTS, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, precision, recall, f1, accuracy, elapsed_time, communication_cost]
    

    try:
        with open(TargetEnc_file_path, "rb+") as pickle_file:
            targetEnc_results = pickle.load(pickle_file)
            targetEnc_results.loc[len(targetEnc_results.index)] = [no_of_training_samples, number_of_test_data, NUM_CLIENTS, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, precision, recall, f1, accuracy, elapsed_time, communication_cost]
            pickle_file.seek(0)
            pickle.dump(targetEnc_results, pickle_file)

    except FileNotFoundError as fnfe:
        with open(TargetEnc_file_path, "wb") as pickle_file:
            pickle.dump(TargetEnc_results_df, pickle_file)

    pickle_file.close()