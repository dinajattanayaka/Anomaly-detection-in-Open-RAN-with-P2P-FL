#!/usr/bin/env python
# coding: utf-8

####################################################################################### Importing Libraries #######################################################################################
import sys
sys.path.append("/scratch/project_2006431/dinaj_files/bin")


import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np
import pandas as pd

import tensorflow as tf
# import tensorflow_federated as tff
from tensorflow.keras import layers
from tensorflow.keras import backend as K

tf.get_logger().setLevel(logging.FATAL)

from category_encoders import TargetEncoder

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


###### Importing Functions #######
from TaregtEncod_Function_Version_7 import TaregtEncod_model
from Clustered_Function_Version_7 import Clustered_model
from Hierarchical_Function_Version_7 import Hierarchical_model
from General_Function_Version_7 import General_model
from Centralized_Function_Version_7 import Centralized_model
from Homomorphic_Function_Version_7 import Homomorphic_model

# from TaregtEncod_Version_5_IID import TaregtEncod_IID
# from Clustered_Version_5_IID import Clustered_IID
# from Hierarchical_Version_5_IID import Hierarchical_IID
# from General_Version_5_IID import General_IID
# from Centralized_Version_5_IID import Centralized_IID

# from TaregtEncod_Version_5_non_IID import TaregtEncod_non_IID
# from Clustered_Version_5_non_IID import Clustered_non_IID
# from Hierarchical_Version_5_non_IID import Hierarchical_non_IID
# from General_Version_5_non_IID import General_non_IID
# from Centralized_Version_5_non_IID import Centralized_non_IID



from datetime import datetime
current_date = datetime.now().date()

print(current_date)




####################################################################################### Custom Functions #######################################################################################


def create_test_data_set(number_of_test_data, Anomaly_data_ratio):
    X_test=[]
    y_test =[]
    Number_of_Anomaly_data_rows = round(number_of_test_data*Anomaly_data_ratio)# create the number of anomaly data rows
    Number_of_Normal_data_rows = number_of_test_data - Number_of_Anomaly_data_rows# deducted from the total number of rows so there are no erros due to rounding off
    Anomaly_index = random.sample(Total_anomalies,Number_of_Anomaly_data_rows)#get a random sample of number of anomaly data rows from the anomaly list
    Normal_index = random.sample(Total_normal_flow,Number_of_Normal_data_rows)#get a random sample of number of anomaly data rows from the normal list
    
    index_list= list(range(0,Full_dataset_len))
    index_list = list(set(index_list)-set(Anomaly_index)-set(Normal_index))
    
    for i in Anomaly_index:
        m=Data_set[i].tolist()
        X_test.append(m)
        y_test.append(Label[i])
    for i in Normal_index:
        m=Data_set[i].tolist()
        X_test.append(m)
        y_test.append(Label[i])
        
    return X_test, y_test, index_list





def create_data_sets_random(num_of_data, Anomaly_data_ratio):
    X_train=[]
    y_train =[]
    
    Number_of_Anomaly_data_rows = math.floor(num_of_data*Anomaly_data_ratio)# create the number of anomaly data rows
    Number_of_Normal_data_rows = num_of_data - Number_of_Anomaly_data_rows# deducted from the total number of rows so there are no erros due to rounding off
    Anomaly_index = random.sample(Available_anomaly_indexes,Number_of_Anomaly_data_rows)#get a random sample of number of anomaly data rows from the anomaly list
    Normal_index = random.sample(Available_normal_indexes,Number_of_Normal_data_rows)#get a random sample of number of anomaly data rows from the normal list
    
    for i in Anomaly_index:
        m=Data_set[i].tolist()
        X_train.append(m)
        y_train.append(Label[i])
    for i in Normal_index:
        m=Data_set[i].tolist()
        X_train.append(m)
        y_train.append(Label[i])
    
    
    
    
    idx_list= list(range(0,len(X_train)))
    NUM_CLIENTS_list= list(range(0,NUM_CLIENTS))
    sample_size=len(X_train)//NUM_CLIENTS
    client_train_dataset_random = collections.OrderedDict()

    
    comb_df = pd.DataFrame(
        {'X': X_train,
         'y': y_train,     
        })
    

    
    for index in range(NUM_CLIENTS):
        client_name = "client_" + str(index)
        client_data_ind = random.sample(idx_list,sample_size)
        idx_list = list(set(idx_list)-set(client_data_ind))
        
        user_df = pd.DataFrame(comb_df.loc[client_data_ind,:])        
        user_df = user_df.sample(frac = 1)
       
        
        user_X = user_df['X'].to_numpy()
        new_dict =np.asarray([user_X[y] for y in range(len(user_X))])
        user_y = user_df['y'].to_numpy()
        new_dict_label = np.asarray([user_y[y] for y in range(len(user_y))])
        client_train_dataset_random[client_name] =collections.OrderedDict((('y', new_dict_label), ('x', new_dict)))
        
    
    return client_train_dataset_random







def create_data_sets_IID(num_of_data, Anomaly_data_ratio):
    X_train=[]
    y_train =[]
    
    Number_of_Anomaly_data_rows = math.floor(num_of_data*Anomaly_data_ratio)# create the number of anomaly data rows
    Number_of_Normal_data_rows = num_of_data - Number_of_Anomaly_data_rows# deducted from the total number of rows so there are no erros due to rounding off
    Anomaly_index = random.sample(Available_anomaly_indexes,Number_of_Anomaly_data_rows)#get a random sample of number of anomaly data rows from the anomaly list
    Normal_index = random.sample(Available_normal_indexes,Number_of_Normal_data_rows)#get a random sample of number of anomaly data rows from the normal list
    
    for i in Anomaly_index:
        m=Data_set[i].tolist()
        X_train.append(m)
        y_train.append(Label[i])
    for i in Normal_index:
        m=Data_set[i].tolist()
        X_train.append(m)
        y_train.append(Label[i])
    
    
    comb_df = pd.DataFrame(
        {'X': X_train,
         'y': y_train,     
        })


    normal_idx_list = comb_df.index[comb_df['y']==0].to_list()
    malicious_idx_list = comb_df.index[comb_df['y']==1].to_list()

    NUM_CLIENTS_list= list(range(0,NUM_CLIENTS))
    sample_size=len(X_train)//NUM_CLIENTS
    client_train_dataset_IID = collections.OrderedDict()

    for user in range(NUM_CLIENTS):
        #get the subset of data for client from database
        #get normal dataset iid
        client_name = "client_" + str(user)
        normal_data_idx = random.sample(normal_idx_list,math.floor(sample_size*(1-anomaly_percentage)))
        normal_idx_list = list(set(normal_idx_list)-set(normal_data_idx))
        normal_df = pd.DataFrame(comb_df.loc[normal_data_idx,:])
        #get malicious dataset iid
        malicious_data_idx = random.sample(malicious_idx_list,math.floor(sample_size*anomaly_percentage))
        malicious_idx_list = list(set(malicious_idx_list)-set(malicious_data_idx))
        malicious_df = pd.DataFrame(comb_df.loc[malicious_data_idx,:])
        #combine malicious and normal datasets
        user_df = pd.concat([normal_df, malicious_df])
        user_df = user_df.sample(frac = 1)
       
        
        user_X = user_df['X'].to_numpy()
        new_dict =np.asarray([user_X[y] for y in range(len(user_X))])
        user_y = user_df['y'].to_numpy()
        new_dict_label = np.asarray([user_y[y] for y in range(len(user_y))])
        client_train_dataset_IID[client_name] =collections.OrderedDict((('y', new_dict_label), ('x', new_dict)))

    return client_train_dataset_IID







def create_data_sets_non_IID(num_of_data, cluster_anomaly_list):
    sample_size = num_of_data//NUM_CLIENTS
    
    non_IID_anomaly_rows = 0
    non_IID_normal_rows = 0
    cluster_anomaly_percentage = {}

    for cluster in range(NUM_CLUSTERS):
        cluster_name = 'cluster_' + str(cluster)
        cluster_anomaly_percentage[cluster_name] = cluster_anomaly_list[cluster]
        non_IID_anomaly_rows += cluster_anomaly_list[cluster]*len(client_clusters[cluster])*sample_size
        non_IID_normal_rows += (1 - cluster_anomaly_list[cluster])*len(client_clusters[cluster])*sample_size

    print(f'Total Anomaly data needed :{non_IID_anomaly_rows}')
    print(f'Total Normal data needed :{non_IID_normal_rows}')

    if int(non_IID_anomaly_rows)> len(Available_anomaly_indexes):
        print("Error : Number of required anomalies is larger than available anomalies")
        return

    if int(non_IID_normal_rows)>len(Available_normal_indexes):
        print("Error : Number of required normal flow is larger than available noraml flows")
        return
    
    
    comb_df = pd.DataFrame(
    {'X': Data_set.tolist(),
     'y': Label.tolist(),     
    })
    
    client_train_dataset_non_IID = collections.OrderedDict()
    
    normal_idx_list = Available_normal_indexes.copy()
    malicious_idx_list = Available_anomaly_indexes.copy()
    
    for cluster in range(NUM_CLUSTERS):
        cluster_name = 'cluster_' + str(cluster)
        cluster_anomaly =  cluster_anomaly_percentage[cluster_name]

        for index in range(len(client_clusters[cluster])):
            client_name = "client_" + str(client_clusters[cluster][index])
            normal_data_idx = random.sample(normal_idx_list,math.floor(sample_size*(1-cluster_anomaly)))
            normal_idx_list = list(set(normal_idx_list)-set(normal_data_idx))
            normal_df = pd.DataFrame(comb_df.loc[normal_data_idx,:])
            
            #get malicious dataset iid
            malicious_data_idx = random.sample(malicious_idx_list,math.floor(sample_size*cluster_anomaly))
            malicious_idx_list = list(set(malicious_idx_list)-set(malicious_data_idx))
            malicious_df = pd.DataFrame(comb_df.loc[malicious_data_idx,:])
            #combine malicious and normal datasets
            user_df = pd.concat([normal_df, malicious_df])
            user_df = user_df.sample(frac = 1)
            
            user_X = user_df['X'].to_numpy()
            new_dict =np.asarray([user_X[y] for y in range(len(user_X))])
            user_y = user_df['y'].to_numpy()
            new_dict_label = np.asarray([user_y[y] for y in range(len(user_y))])
            client_train_dataset_non_IID[client_name] =collections.OrderedDict((('y', new_dict_label), ('x', new_dict)))
    
    return client_train_dataset_non_IID, cluster_anomaly_percentage














if __name__ == '__main__':
    
    print('\n')
    print("File names ends with: _17112022_batches")
    print('\n')
    
    ####################################################################################### Clustering  #######################################################################################    

    ## Initialization
    NUM_CLIENTS = 10
    NUM_CLUSTERS = 3


    ## Randomize locatios of the clients
    location_list = []
    for user in range(NUM_CLIENTS):
        location_list.append([random.randint(1,500),random.randint(1,500) ])

    gnb_loc = pd.DataFrame(location_list, columns=['x','y'])

    gnb_loc_ls = gnb_loc.to_numpy()


    


    dist_dic = {}

    for index in range(NUM_CLIENTS):
        client_name = "client_" + str(index)
        tmp_ls = []
        for itr in range(NUM_CLIENTS):
            tmp_ls.append(np.linalg.norm(gnb_loc_ls[index,:]-gnb_loc_ls[itr,:]))
        dist_dic[client_name] = tmp_ls


    
    ## K-means clustering
    kmeans = KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(gnb_loc[['x','y']])

    cluster_centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    print('\n')
    print(f'cluster_centroids : {cluster_centroids}')
    print('\n')
    print(f'cluster_labels{cluster_labels}')


    # ## Ploting the clients locations and cluster means
    # plt.scatter(gnb_loc['x'], gnb_loc['y'], s=50, c='blue')
    # plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='red', s=100, alpha=0.5, marker='x')


    
    ## Labeling the clients
    client_labels = {}

    for index in range(NUM_CLIENTS):
        client_name = "client_" + str(index)
        client_labels[client_name] = cluster_labels[index]

    client_clusters = {}
    for itr in range(NUM_CLUSTERS):
        client_clusters[itr] = np.array(np.where(cluster_labels == itr))
        client_clusters[itr] = client_clusters[itr][0]


    print('\n')
    print(f'client_clusters: {client_clusters}')





    ####################################################################################### Hierarchical parameter sharing #######################################################################################
    resource_dic = {}

    for index in range(NUM_CLIENTS):
        client_name = "client_" + str(index)
        resource_dic[client_name] =  random.uniform(0., 100.)

    resource_lead = {}
    resource_lead_mod = {}
    cluster_weights_avg = {}
    len_avg = 0

    for cluster in range(NUM_CLUSTERS):
        cluster_name = "cluster_" + str(cluster)
        len_avg += len(client_clusters[cluster])
        tmp_val = 0
        for index in range(len(client_clusters[cluster])):
            client_name = "client_" + str(client_clusters[cluster][index])  
            model_name = "model_" + str(client_clusters[cluster][index]) 

            if resource_dic[client_name] > tmp_val:
                tmp_val = resource_dic[client_name]
                resource_lead[cluster_name] = client_name
                resource_lead_mod[cluster_name] = model_name

    for cluster in range(NUM_CLUSTERS):
        cluster_name = "cluster_" + str(cluster)
        cluster_weights_avg[cluster_name] = len(client_clusters[cluster])/len_avg


    print('\n')
    print(f'resource_lead: {resource_lead}')
    print('\n')
    print(f'cluster_weights_avg: {cluster_weights_avg}')






    ####################################################################################### Loading the dataset #######################################################################################
    data_all = pd.read_csv("UNSW_NB15_training_testing_concat.csv")
    data = data_all
    # data.head()

    print('\n')
    print(f'{data.label.value_counts()}')


    print('\n')
    print(f'{data.attack_cat.value_counts()}')




    ## Format data

    # Drop id as it is not relevant
    data.drop(columns=['id'], axis=1, inplace=True)

    # Drop attack_cat as it is not relevant
    data.drop(['attack_cat'], axis=1, inplace=True)

    # data.shape
    # data.info()


    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]


    
    # Convert to cetegorical columns
    categorical_cols = ["proto",
                         "service",
                         "state"]

    encoder = TargetEncoder()
    encoded = encoder.fit_transform(X[categorical_cols],y)

    # Update data with new columns
    X.drop(categorical_cols, axis=1, inplace=True)
    X = pd.concat([encoded, X], axis=1)

    # encoded






    ## Normalize the data
    scaler = MinMaxScaler()
    X_array = scaler.fit_transform(X)

    X = pd.DataFrame(X_array, columns=X.columns)

    # X.head()


    # dataset_bin = pd.concat([X,y],axis=1)






    
    ####################################################################################### Split the dataset #######################################################################################
    Data_set = np.array(X)
    Label = np.array(y)



    Full_dataset_len = len(X)
    print('\n')
    print('\n')
    print(f'Full_dataset_len: {Full_dataset_len}')


    Total_anomalies= list()
    Total_normal_flow = list()
    Total_index_list= list(range(0,Full_dataset_len))


    for i in Total_index_list:
        if Label[i] ==1:
            Total_anomalies.append(i)
        else:
            Total_normal_flow.append(i)



    ##### Create the dest dataset
    number_of_test_data=10000;
    test_anomaly_percentage = 0.6
    
    X_test, y_test, index_list = create_test_data_set(number_of_test_data, test_anomaly_percentage)

    print('There are %d samples of data for training.' % (Full_dataset_len-number_of_test_data))


    #### Remaining data
    Available_anomaly_indexes= list()
    Available_normal_indexes = list()
    Length_of_dataset = len(index_list) # take the length of the remaining data list
    for i in index_list:
        if Label[i] ==1:
            Available_anomaly_indexes.append(i)
        else:
            Available_normal_indexes.append(i)

    print("Number of Anomalies should be less than %d" %(len(Available_anomaly_indexes)))
    print("Number of Normal flow should be less than %d" %(len(Available_normal_indexes)))









    ####################################################################################### Parameters #######################################################################################

    no_of_training_samples = 200000  # Number of samples used 
    anomaly_percentage = 0.6

    # Manual entry
#     cluster_anomaly_list = [0.6, 0.6, 0.6, 0.6, 0.6]
#     cluster_anomaly_list = [0.6, 0.5, 0.4, 0.5, 0.6]
#     cluster_anomaly_list = [0.6, 0.5, 0.4, 0.7, 0.6]
    cluster_anomaly_list = [0.6, 0.6, 0.6]



    BATCH_SIZE = 100
    # BATCH_SIZE_ls = [100]
#     BATCH_SIZE_ls = list(range(20, 2010, 20))
    BATCH_SIZE_ls = list(range(100, 20010, 100))

    PREFETCH_BUFFER = 10

    NUM_EPOCHS = 10
    # NUM_EPOCHS_ls = list(range(10, 101, 10))

    NUM_ROUNDS = 50
#     NUM_ROUNDS_ls=list(range(2, 81,1))
#     NUM_ROUNDS_ls = [5]

    SHARE_INTERVAL = 5




    #### Filepaths to save the results

    random_TargetEnc_file_path = "random/random_TargetEnc_17112022_batches.pkl"
    random_Clustered_file_path = "random/random_Clustered_17112022_batches.pkl"
    random_Hierarchical_file_path = "random/random_Hierarchical_17112022_batches.pkl"
    random_General_file_path = "random/random_General_17112022_batches.pkl"
    random_Centralized_file_path = "random/random_Centralized_17112022_batches.pkl"
    random_Homomorphic_file_path = "random/random_Homomorphic_17112022_batches.pkl"

    IID_TargetEnc_file_path = "IID/IID_TargetEnc_17112022_batches.pkl"
    IID_Clustered_file_path = "IID/IID_Clustered_17112022_batches.pkl"
    IID_Hierarchical_file_path = "IID/IID_Hierarchical_17112022_batches.pkl"
    IID_General_file_path = "IID/IID_General_17112022_batches.pkl"
    IID_Centralized_file_path = "IID/IID_Centralized_17112022_batches.pkl"
    IID_Homomorphic_file_path = "IID/IID_Homomorphic_17112022_batches.pkl"


    non_IID_TargetEnc_file_path = "non_IID/non_IID_TargetEnc_17112022_batches.pkl"
    non_IID_Clustered_file_path = "non_IID/non_IID_Clustered_17112022_batches.pkl"
    non_IID_Hierarchical_file_path = "non_IID/non_IID_Hierarchical_17112022_batches.pkl"
    non_IID_General_file_path = "non_IID/non_IID_General_17112022_batches.pkl"
    non_IID_Centralized_file_path = "non_IID/non_IID_Centralized_17112022_batches.pkl"
    non_IID_Homomorphic_file_path = "non_IID/non_IID_Homomorphic_17112022_batches.pkl"
    
    
    
    


    ####################################################################################### Creating the training datasets #######################################################################################

    client_train_dataset_random = create_data_sets_random(no_of_training_samples,anomaly_percentage)
    client_train_dataset_IID = create_data_sets_IID(no_of_training_samples,anomaly_percentage)
    client_train_dataset_non_IID, cluster_anomaly_percentage = create_data_sets_non_IID(no_of_training_samples,cluster_anomaly_list)


    ## Calculate the total anomalies
    random_anomaly_sum = 0
    random_anomaly_dic = {}
    sample_size = no_of_training_samples//NUM_CLIENTS

    for index in range(NUM_CLIENTS):
        client_name = "client_" + str(index)
        anomaly_count = np.count_nonzero(client_train_dataset_random[client_name]['y'] == 1)
        random_anomaly_sum += anomaly_count
        random_anomaly_dic[client_name] = anomaly_count/sample_size


    random_anomaly_dic['Total'] = random_anomaly_sum/(no_of_training_samples)
    print(f'Total Anomalies are : {random_anomaly_sum}')
 

    # for index in range(NUM_CLIENTS):
    #     client_name = "client_" + str(index)
    #     display(np.count_nonzero(client_train_dataset_IID[client_name]['y'] == 1))


    # for cluster in range(NUM_CLUSTERS):
    #     cluster_name = 'cluster_' + str(cluster)
    #     print(f'Considering : {cluster_name}')
    #     for index in range(len(client_clusters[cluster])):
    #         client_name = "client_" + str(client_clusters[cluster][index])
    #         display(np.count_nonzero(client_train_dataset_non_IID[client_name]['y'] == 1))


    print('\n')
    print(f'Cluster anomaly percentages : {cluster_anomaly_percentage}')
    print('\n')
    print(f'Random anomaly percentages : {random_anomaly_dic}')



    # def create_keras_model():
    #   return tf.keras.models.Sequential([
    #   tf.keras.layers.InputLayer(input_shape=(42, )),
    #   tf.keras.layers.Dense(30,activation='relu'),
    #   tf.keras.layers.Dense(10,activation='relu'),
    #   tf.keras.layers.Dense(2),
    #   tf.keras.layers.Softmax(),
    #   ])


    
    
    
    
    

    ####################################################################################### Federated Learning #######################################################################################
    for BATCH_SIZE in BATCH_SIZE_ls:
        print('\n')
        print(f'Starting the cycle of batch size: {BATCH_SIZE}') 


#         ##################### Randomly distributed data #####################
#         print('\n')
#         print(f'############################### Considering random dataset ###############################') 
#         print('\n')
#         print(f'>>>>>>>> Starting random TaregtEncod <<<<<<<<')
#         print('\n')
#         TaregtEncod_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_random, random_TargetEnc_file_path)
#         print('\n')
#         print(f'>>>>>>>> Starting random Clustered <<<<<<<<')
#         print('\n')
#         Clustered_model(NUM_CLIENTS, NUM_CLUSTERS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_clusters, client_train_dataset_random, random_Clustered_file_path)
#         print('\n')
#         print(f'>>>>>>>> Starting random Hierarchical <<<<<<<<')
#         print('\n')
#         Hierarchical_model(NUM_CLIENTS, NUM_CLUSTERS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic,  BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, SHARE_INTERVAL, X_test, y_test, client_clusters, client_train_dataset_random, resource_lead_mod, cluster_weights_avg, random_Hierarchical_file_path)
#         print('\n')
#         print(f'>>>>>>>> Starting random General <<<<<<<<')
#         print('\n')
#         General_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_random, random_General_file_path)
#         print('\n')
#         print(f'>>>>>>>> Starting random Centralized <<<<<<<<')
#         print('\n')
#         Centralized_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic,  BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_random, random_Centralized_file_path)



        
        #################### IID data #####################
        print('\n')
        print(f'############################### Considering IID dataset ###############################') 
        print('\n')
        print(f'>>>>>>>> Starting IID TargetEncod <<<<<<<<')
        print('\n')
        TaregtEncod_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_IID, IID_TargetEnc_file_path)
        print('\n')
        print(f'>>>>>>>> Starting IID Clustered <<<<<<<<')
        print('\n')
        Clustered_model(NUM_CLIENTS, NUM_CLUSTERS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_clusters, client_train_dataset_IID, IID_Clustered_file_path)
        print('\n')
        print(f'>>>>>>>> Starting IID Hierarchical <<<<<<<<')
        print('\n')
        Hierarchical_model(NUM_CLIENTS, NUM_CLUSTERS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic,  BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, SHARE_INTERVAL, X_test, y_test, client_clusters, client_train_dataset_IID, resource_lead_mod, cluster_weights_avg, IID_Hierarchical_file_path)
        print('\n')
        print(f'>>>>>>>> Starting IID General <<<<<<<<')
        print('\n')
        General_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_IID, IID_General_file_path)
        print('\n')
        print(f'>>>>>>>> Starting IID Centralized <<<<<<<<')
        print('\n')
        Centralized_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic,  BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_IID, IID_Centralized_file_path)





#         ##################### non IID data #####################
#         print('\n')
#         print(f'############################### Considering Non-IID dataset ###############################') 
#         print('\n')
#         print(f'>>>>>>>> Starting Non-IID TaregtEncod <<<<<<<<')
#         print('\n')
#         TaregtEncod_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_non_IID, non_IID_TargetEnc_file_path)
#         print('\n')
#         print(f'>>>>>>>> Starting Non-IID Clustered <<<<<<<<')
#         print('\n')
#         Clustered_model(NUM_CLIENTS, NUM_CLUSTERS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_clusters, client_train_dataset_non_IID, non_IID_Clustered_file_path)
#         print('\n')
#         print(f'>>>>>>>> Starting Non-IID Hierarchical <<<<<<<<')
#         print('\n')
#         Hierarchical_model(NUM_CLIENTS, NUM_CLUSTERS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic,  BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, SHARE_INTERVAL, X_test, y_test, client_clusters, client_train_dataset_non_IID, resource_lead_mod, cluster_weights_avg, non_IID_Hierarchical_file_path)
#         print('\n')
#         print(f'>>>>>>>> Starting Non-IID General <<<<<<<<')
#         print('\n')
#         General_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic, BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_non_IID, non_IID_General_file_path)
#         print('\n')
#         print(f'>>>>>>>> Starting Non-IID Centralized <<<<<<<<')
#         print('\n')
#         Centralized_model(NUM_CLIENTS, no_of_training_samples, number_of_test_data, anomaly_percentage, cluster_anomaly_percentage, random_anomaly_dic,  BATCH_SIZE, NUM_EPOCHS, NUM_ROUNDS, X_test, y_test, client_train_dataset_non_IID, non_IID_Centralized_file_path)

        print('\n')
