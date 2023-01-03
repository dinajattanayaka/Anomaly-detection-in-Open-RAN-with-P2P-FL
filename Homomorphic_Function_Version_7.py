#!/usr/bin/env python
# coding: utf-8

def Homomorphic_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset, Homomorphic_file_path):

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
    
    
    import phe
    from phe import paillier
    from numba import jit
    import sympy

    

    ####### Important ######
    np.random.seed(0)    


    ####### Import custom functions #######    
    from worker_functions import init_model_train
    from worker_functions import iterative_model_fit
    from worker_functions import model_prediction_targetEnc

    from worker_functions import encryption_func
    from worker_functions import partital_decrypt_func
    from worker_functions import ThresholdPaillier
    from worker_functions import combineShares
    
    
    ####### Initalize the results dataframe
    Homomorphic_results_df = pd.DataFrame({
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


    


#     ####################################################################################### Custom functions #######################################################################################
#     def encryption_func(index, multi_encrypted_prev_dic, prev_dic, dim_list, public_key):
#         model_name = "model_" + str(index)
#     #     encrypted_prev_dic[model_name] = []

#         multi_val_temp_list = []

#         for layer in range(len(dim_list)):
#             layer_list = []
#             for itr1 in range(2):
#                 tmp_list = []
#                 if itr1==0:
#                     for itr2 in range(dim_list[layer][itr1][0]):
#                         w_list = []
#                         for itr3 in range(dim_list[layer][itr1][1]):
#                             enc_val = public_key.encrypt(int(prev_dic[model_name][layer][itr1][itr2][itr3]*1e15))
#                             w_list.append(enc_val)
#                         tmp_list.append(w_list)
#                 else:
#                     for itr2 in range(dim_list[layer][itr1]):
#                         enc_val = public_key.encrypt(int(prev_dic[model_name][layer][itr1][itr2]*1e15))
#                         tmp_list.append(enc_val)
#                 layer_list.append(tmp_list)
#             multi_val_temp_list.append(layer_list)

#         multi_encrypted_prev_dic[model_name] = multi_val_temp_list.copy()




#     def partital_decrypt_func(index, multi_partial_decrypt_dic, encrypted_sum_dic, dim_list, private_key_dic):   
#         model_name = "model_" + str(index)

#         model_priv_key = private_key_dic[model_name]

#         temp_partial_decrypt_list = []

#         for layer in range(len(dim_list)):
#             tmp = []
#             for itr1 in range(2):
#                 tmp2 = []
#                 if itr1==0:
#                     for i in range(dim_list[layer][itr1][0]):
#                         tmp2.append([])
#                 else:
#                     tmp2 = []
#                 tmp.append(tmp2)
#             temp_partial_decrypt_list.append(tmp)



#         for layer in range(len(dim_list)):
#             for itr1 in range(2):
#                 if itr1 == 0:
#                     for itr2 in range(dim_list[layer][itr1][0]):
#                         for itr3 in range(dim_list[layer][itr1][1]):                  
#                             temp_partial_decrypt_list[layer][itr1][itr2].append(model_priv_key.partialDecrypt(encrypted_sum_dic[model_name][layer][itr1][itr2][itr3]))
#                 else:
#                     for itr2 in range(dim_list[layer][itr1]):
#                         temp_partial_decrypt_list[layer][itr1].append(model_priv_key.partialDecrypt(encrypted_sum_dic[model_name][layer][itr1][itr2]))

#         multi_partial_decrypt_dic[model_name] = temp_partial_decrypt_list.copy()



#     class ThresholdPaillier(object):
#         def __init__(self,size_of_n):
#             #size_of_n = 1024
#             pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
#             self.p1 = priv.p
#             self.q1 = priv.q

#             while sympy.isprime(2*self.p1 +1)!= True:
#                 pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
#                 self.p1 = priv.p
#             while sympy.isprime(2*self.q1 +1)!= True:
#                 pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
#                 self.q1 = priv.q

#             self.p = (2*self.p1) + 1
#             self.q = (2*self.q1) + 1
#             print(sympy.isprime(self.p),sympy.isprime(self.q),sympy.isprime(self.p1),sympy.isprime(self.q1))
#             self.n = self.p * self.q
#             self.s = 1
#             self.ns = pow(self.n, self.s)
#             self.nSPlusOne = pow(self.n,self.s+1)
#             self.nPlusOne = self.n + 1
#             self.nSquare = self.n*self.n

#             self.m = self.p1 * self.q1
#             self.nm = self.n*self.m
#             self.l = NUM_CLIENTS # Number of shares of private key
#             self.w = NUM_CLIENTS # The minimum of decryption servers needed to make a correct decryption.
#             self.delta = self.factorial(self.l)
#             self.rnd = random.randint(1,1e50)
#             self.combineSharesConstant = sympy.mod_inverse((4*self.delta*self.delta)%self.n, self.n)
#             self.d = self.m * sympy.mod_inverse(self.m, self.n)

#             self.ais = [self.d]
#             for i in range(1, self.w):
#                 self.ais.append(random.randint(0,self.nm-1))

#             self.r = random.randint(1,self. p) ## Need to change upper limit from p to one in paper
#             while math.gcd(self.r,self.n) != 1:
#                 self.r = random.randint(0, self.p)
#             self.v = (self.r*self.r) % self.nSquare

#             self.si = [0] * self.l
#             self.viarray = [0] * self.l

#             for i in range(self.l):
#                 self.si[i] = 0
#                 X = i + 1
#                 for j in range(self.w):
#                     self.si[i] += self.ais[j] * pow(X, j)
#                 self.si[i] = self.si[i] % self.nm
#                 self.viarray[i] = pow(self.v, self.si[i] * self.delta, self.nSquare)

#             self.priv_keys = []
#             for i in range(self.l):
#                 self.priv_keys.append(ThresholdPaillierPrivateKey(self.n, self.l, self.combineSharesConstant, self.w, \
#                                                 self.v, self.viarray, self.si[i], i+1, self.r, self.delta, self.nSPlusOne))
#             self.pub_key = ThresholdPaillierPublicKey(self.n, self.nSPlusOne, self.r, self.ns, self.w,\
#                                                      self.delta, self.combineSharesConstant)

#         def factorial(self, n):
#             fact = 1
#             for i in range(1,n+1):
#                 fact *= i
#             return fact

#         def computeGCD(self, x, y):
#            while(y):
#                x, y = y, x % y
#            return x

#     class PartialShare(object):
#         def __init__(self, share, server_id):
#             self.share = share
#             self.server_id =server_id

#     class ThresholdPaillierPrivateKey(object):
#         def __init__(self,n, l,combineSharesConstant, w, v, viarray, si, server_id, r, delta, nSPlusOne):
#             self.n = n
#             self.l = l
#             self.combineSharesConstant = combineSharesConstant
#             self.w = w
#             self.v = v
#             self.viarray = viarray
#             self.si = si
#             self.server_id = server_id
#             self.r = r
#             self.delta = delta
#             self.nSPlusOne = nSPlusOne

#         def partialDecrypt(self, c):
#             return PartialShare(pow(c.c, self.si*2*self.delta, self.nSPlusOne), self.server_id)

#     class ThresholdPaillierPublicKey(object):
#         def __init__(self,n, nSPlusOne, r, ns, w, delta, combineSharesConstant):
#             self.n = n
#             self.nSPlusOne = nSPlusOne
#             self.r = r
#             self.ns =ns
#             self.w = w
#             self.delta = delta
#             self.combineSharesConstant = combineSharesConstant

#         def encrypt(self, msg):
#             msg = msg % self.nSPlusOne if msg < 0 else msg
#             c = (pow(self.n+1, msg, self.nSPlusOne) * pow(self.r, self.ns, self.nSPlusOne)) % self.nSPlusOne
#             return EncryptedNumber(c, self.nSPlusOne, self.n)

#     class EncryptedNumber(object):
#         def __init__(self, c, nSPlusOne, n):
#             self.c = c
#             self.nSPlusOne = nSPlusOne
#             self.n = n

#         def __mul__(self, cons):
#             if cons < 0:
#                 return EncryptedNumber(pow(sympy.mod_inverse(self.c, self.nSPlusOne), -cons, self.nSPlusOne), self.nSPlusOne, self.n)
#             else:
#                 return EncryptedNumber(pow(self.c, cons, self.nSPlusOne), self.nSPlusOne, self.n)

#         def __add__(self, c2):
#             return EncryptedNumber((self.c * c2.c) % self.nSPlusOne, self.nSPlusOne, self.n)

#     def combineShares(shrs, w, delta, combineSharesConstant, nSPlusOne, n, ns):
#             cprime = 1
#             for i in range(w):
#                 ld = delta
#                 for iprime in range(w):
#                     if i != iprime:
#                         if shrs[i].server_id != shrs[iprime].server_id:
#                             ld = (ld * -shrs[iprime].server_id) // (shrs[i].server_id - shrs[iprime].server_id)
#                 #print(ld)
#                 shr = sympy.mod_inverse(shrs[i].share, nSPlusOne) if ld < 0 else shrs[i].share
#                 ld = -1*ld if ld <1 else ld
#                 temp = pow(shr, 2 * ld, nSPlusOne)
#                 cprime = (cprime * temp) % nSPlusOne
#             L = (cprime - 1) // n
#             result = (L * combineSharesConstant) % n
#             return result - ns if result > (ns // 2) else result


        
        
        

    ################### Encryption key generation ###################
    tp = ThresholdPaillier(1024)

    priv_keys_list = tp.priv_keys
    public_key = tp.pub_key

    
    
    
    
    ###################################################################################### Start the learning ######################################################################################
    start_time = time.time()


    accuracy_dic = {}
    loss_dic = {}

    communication_cost = {}

    time_calc_dic = {}

    time_calc_dic['Encryption'] = []
    time_calc_dic['Encrypt_sum_cal'] = []
    time_calc_dic['Partital_decrypt'] = []
    time_calc_dic['Final_decrypt_avg'] = []
    time_calc_dic['Model_train'] = []
    time_calc_dic['Final_sync'] = []


    for index in range(NUM_CLIENTS):
        model_name = "model_" + str(index)
        communication_cost[model_name] = 0

    communication_cost['Aggregator'] = 0 


    ##################### Sending global public key and secret key shares ###################
    private_key_dic = {}
    for index in range(NUM_CLIENTS):
        model_name = "model_" + str(index)
        private_key_dic[model_name] = priv_keys_list[index]

    # Sendig private key share and public key securely
    communication_cost['Aggregator'] = communication_cost['Aggregator'] + NUM_CLIENTS





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


    ## For hierarhical and centralized methods
    total_model_parameters = (layer1_input_size * layer1_output_size + layer1_bias_size) + (layer2_input_size * layer2_output_size + layer2_bias_size) + (layer3_input_size * layer3_output_size + layer3_bias_size)  






    ####################################################################################### Iterative training #######################################################################################


    for round_num in range(NUM_ROUNDS):

        print(f"The round: {round_num + 1}")

        accuracy_dic[round_num] = {}
        loss_dic[round_num] = {}


        ######################################## Encrpting the parameters ########################################   
        encrypt_time = time.time()
    #     print(f'Encryption of the intial weights values')
        with Manager() as manager:
            multi_encrypted_prev_dic = manager.dict()
            pool = Pool()
            pool.starmap(encryption_func, zip(range(NUM_CLIENTS), repeat(multi_encrypted_prev_dic), repeat(prev_dic), repeat(dim_list), repeat(public_key)))
            pool.close()
            pool.join()  

            encrypted_prev_dic = multi_encrypted_prev_dic.copy()



        encrypt_elap = time.time() - encrypt_time
        time_calc_dic['Encryption'].append(encrypt_elap)




        ######################################## Calculating Encrypted sums ########################################      
        calc_time = time.time()

        encrypted_sum_dic = {}

        encrypted_aggr_list= []
        for layer in range(len(dim_list)):
            tmp = []
            for itr1 in range(2):
                tmp2 = []
                if itr1==0:
                    for i in range(dim_list[layer][itr1][0]):
                        tmp2.append([])
                else:
                    tmp2 = []
                tmp.append(tmp2)
            encrypted_aggr_list.append(tmp)


        for layer in range(len(dim_list)):
            for itr1 in range(2):
                if itr1 == 0:
                    for itr2 in range(dim_list[layer][itr1][0]):    
                        for itr3 in range(dim_list[layer][itr1][1]):
                            sum_calc = encrypted_prev_dic['model_0'][layer][itr1][itr2][itr3]
                            for itr4 in range(1, NUM_CLIENTS):
                                temp_name = "model_" + str(itr4)
                                sum_calc += encrypted_prev_dic[temp_name][layer][itr1][itr2][itr3]

                            encrypted_aggr_list[layer][itr1][itr2].append(sum_calc)



                else:
                    for itr2 in range(dim_list[layer][itr1]):        
                        sum_calc = encrypted_prev_dic['model_0'][layer][itr1][itr2]
                        for itr3 in range(1, NUM_CLIENTS):
                            temp_name = "model_" + str(itr3)
                            sum_calc += encrypted_prev_dic[temp_name][layer][itr1][itr2]


                        encrypted_aggr_list[layer][itr1].append(sum_calc)




        ## Copying the values to each model
        for user in range(NUM_CLIENTS):
            model_name = "model_" + str(user)
            encrypted_sum_dic[model_name] = copy.deepcopy(encrypted_aggr_list)

            communication_cost[model_name] = communication_cost[model_name] + total_model_parameters * (NUM_CLIENTS - 1)


        calc_elap = time.time() - calc_time
        time_calc_dic['Encrypt_sum_cal'].append(calc_elap)   





        ######################################## Partital decryption ########################################   
        partital_decrypt_time = time.time()
    #     print(f'Partital decryption')
        with Manager() as manager:
            multi_partial_decrypt_dic = manager.dict()
            pool = Pool()
            pool.starmap(partital_decrypt_func, zip(range(NUM_CLIENTS), repeat(multi_partial_decrypt_dic), repeat(encrypted_sum_dic), repeat(dim_list), repeat(private_key_dic)))
            pool.close()
            pool.join() 

            partial_decrypt_dic = multi_partial_decrypt_dic.copy()

        partital_decrypt_elap = time.time() - partital_decrypt_time    

        time_calc_dic['Partital_decrypt'].append(partital_decrypt_elap) 






        ######################################## Full decryption and average calculation ########################################    
        decrypt_avg_time = time.time()

        final_dic = {}

        decrypted_avg_list= []
        for layer in range(len(dim_list)):
            tmp = []
            for itr1 in range(2):
                tmp2 = []
                if itr1==0:
                    for i in range(dim_list[layer][itr1][0]):
                        tmp2.append([])
                else:
                    tmp2 = []
                tmp.append(tmp2)
            decrypted_avg_list.append(tmp)


        for layer in range(len(dim_list)):
            for itr1 in range(2):
                if itr1 == 0:
                    for itr2 in range(dim_list[layer][itr1][0]):    
                        for itr3 in range(dim_list[layer][itr1][1]):
                            temp_share_list = []
                            for itr4 in range(0, NUM_CLIENTS):
                                temp_name = "model_" + str(itr4)
                                temp_share_list.append(partial_decrypt_dic[temp_name][layer][itr1][itr2][itr3])
                            decrypt_val = combineShares(temp_share_list, public_key.w, public_key.delta, public_key.combineSharesConstant, public_key.nSPlusOne, public_key.n, public_key.ns)
                            decrypted_avg_list[layer][itr1][itr2].append(decrypt_val/(1e15*NUM_CLIENTS))


                else:
                    for itr2 in range(dim_list[layer][itr1]):        
                        temp_share_list = []
                        for itr3 in range(0, NUM_CLIENTS):
                            temp_name = "model_" + str(itr3)
                            temp_share_list.append(partial_decrypt_dic[temp_name][layer][itr1][itr2])

                        decrypt_val = combineShares(temp_share_list, public_key.w, public_key.delta, public_key.combineSharesConstant, public_key.nSPlusOne, public_key.n, public_key.ns)
                        decrypted_avg_list[layer][itr1].append(decrypt_val/(1e15*NUM_CLIENTS))




        ## copy values
        for user in range(NUM_CLIENTS):
            model_name = "model_" + str(user)
            prev_dic[model_name] = copy.deepcopy(decrypted_avg_list)
            final_dic[model_name] = copy.deepcopy(decrypted_avg_list)
            communication_cost[model_name] = communication_cost[model_name] + total_model_parameters * (NUM_CLIENTS - 1) 


        # Make final_dic a numpy array
        for user in range(NUM_CLIENTS):
            model_name = "model_" + str(user)
            for layer in range(len(dim_list)):
                for itr1 in range(2):
                    final_dic[model_name][layer][itr1] = np.array(final_dic[model_name][layer][itr1])  
            final_dic[model_name] = np.array(final_dic[model_name], dtype=object) 



        decrypt_avg_elap = time.time() - decrypt_avg_time

        time_calc_dic['Final_decrypt_avg'].append(decrypt_avg_elap) 




        ######################################## New model with parameter averages ########################################         

        nm_time = time.time() 

        with Manager() as manager:
            itr_perv_dic = manager.dict()
            pool = Pool() 

            pool.starmap(iterative_model_fit, zip(range(NUM_CLIENTS), repeat(itr_perv_dic), repeat(client_train_dataset), repeat(NUM_EPOCHS), repeat(BATCH_SIZE), repeat(final_dic)))
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

    elapsed_time = time.time() - start_time 
    training_time = str(timedelta(seconds=elapsed_time))




    #### Time calculation ####
    encrypt_elap_tot = sum(time_calc_dic['Encryption'])
    encrypt_sum_elap_tot = sum(time_calc_dic['Encrypt_sum_cal']) 
    partial_decrypt_elap_tot = sum(time_calc_dic['Partital_decrypt']) 
    final_decrypt_avg_elap_tot = sum(time_calc_dic['Final_decrypt_avg'])
    nm_elap_tot = sum(time_calc_dic['Model_train'])
    ad_elap_tot = sum(time_calc_dic['Final_sync']) 

    print(f'Initial model training time - {str(timedelta(seconds= init_elap_time))}')
    print(f'Parameter encryption time - {str(timedelta(seconds=encrypt_elap_tot))}')
    print(f'Encrypted sum calculation time - {str(timedelta(seconds=encrypt_sum_elap_tot))}')
    print(f'Partital decryption time - {str(timedelta(seconds=partial_decrypt_elap_tot))}')
    print(f'Full decryption and average calculation time - {str(timedelta(seconds=final_decrypt_avg_elap_tot))}')
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
    Homomorphic_results_df.loc[len(Homomorphic_results_df.index)] = [no_of_training_samples, number_of_test_data, NUM_CLIENTS, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, precision, recall, f1, accuracy, elapsed_time, communication_cost]
    

    try:
        with open(Homomorphic_file_path, "rb+") as pickle_file:
            homomorphic_results = pickle.load(pickle_file)
            homomorphic_results.loc[len(homomorphic_results.index)] = [no_of_training_samples, number_of_test_data, NUM_CLIENTS, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, precision, recall, f1, accuracy, elapsed_time, communication_cost]
            pickle_file.seek(0)
            pickle.dump(homomorphic_results, pickle_file)

    except FileNotFoundError as fnfe:
        with open(Homomorphic_file_path, "wb") as pickle_file:
            pickle.dump(Homomorphic_results_df, pickle_file)

    pickle_file.close()

