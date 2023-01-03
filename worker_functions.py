
import numpy as np


import tensorflow as tf
# import tensorflow_federated as tff
from tensorflow.keras import layers
from tensorflow.keras import backend as K

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
# import numpy as np
import random
from numba import jit
import sympy
import math


NUM_CLIENTS = 100
alpha = 0.001


#####################################
def create_keras_model():
  return tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=(42, )),
  tf.keras.layers.Dense(30,activation='relu'),
  tf.keras.layers.Dense(10,activation='relu'),
  tf.keras.layers.Dense(2),
  tf.keras.layers.Softmax(),
  ])


#####################################
def init_model_train(index, multi_prev_dic, client_train_dataset, NUM_EPOCHS, BATCH_SIZE):
    
    model_name = "model_"  + str(index)
    client_name = "client_"  + str(index)
#     print(f'Starting the initial training of: {model_name}')
    init_model = create_keras_model()
    
    optim = tf.keras.optimizers.Adam(learning_rate=alpha)
    init_model.compile(optimizer=optim,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    init_model.fit(client_train_dataset[client_name]['x'], client_train_dataset[client_name]['y'], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,verbose=0)

    temp_ls = []

    for layer in range(len(init_model.layers)-1):
        temp_ls.append([init_model.layers[layer].get_weights()[0], init_model.layers[layer].get_weights()[1]])
     
    multi_prev_dic[model_name] = np.array(temp_ls, dtype=object)
    
    
    tf.keras.backend.clear_session()


    
    
    
    
    
    
#####################################    
def iterative_model_fit(index, itr_perv_dic, client_train_dataset, NUM_EPOCHS, BATCH_SIZE, prev_dic):
    

    model_name = "model_"  + str(index)
    client_name = "client_"  + str(index)
    
    model = create_keras_model()
    for layer in range(len(model.layers)-1):
           model.layers[layer].set_weights([prev_dic[model_name][layer][0], prev_dic[model_name][layer][1]])


#     kernel_initializer_a1= tf.keras.initializers.constant(prev_dic[model_name][0][0])
#     bias_initializer_a1=tf.keras.initializers.constant(prev_dic[model_name][0][1])
#     kernel_initializer_b1= tf.keras.initializers.constant(prev_dic[model_name][1][0])
#     bias_initializer_b1=tf.keras.initializers.constant(prev_dic[model_name][1][1])
#     kernel_initializer_c1= tf.keras.initializers.constant(prev_dic[model_name][2][0])
#     bias_initializer_c1=tf.keras.initializers.constant(prev_dic[model_name][2][1])


#     model = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(input_shape=(42, )),
#         tf.keras.layers.Dense(30,activation='relu',kernel_initializer=kernel_initializer_a1,bias_initializer=bias_initializer_a1),
#         tf.keras.layers.Dense(10,activation='relu',kernel_initializer=kernel_initializer_b1,bias_initializer=bias_initializer_b1),
#         tf.keras.layers.Dense(2,kernel_initializer=kernel_initializer_c1,bias_initializer=bias_initializer_c1),
#         tf.keras.layers.Softmax()
#     ])

    optim = tf.keras.optimizers.Adam(learning_rate=alpha)
    model.compile(optimizer=optim,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(client_train_dataset[client_name]['x'], client_train_dataset[client_name]['y'], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,verbose=0)
    
    temp_ls =[]
    for layer in range(len(model.layers)-1):
        temp_ls.append([model.layers[layer].get_weights()[0], model.layers[layer].get_weights()[1]])

    
    itr_perv_dic[model_name] = np.array(temp_ls, dtype=object)
    
    tf.keras.backend.clear_session()

    
    

    
    
#####################################    
def model_prediction_targetEnc(index, multi_predict_dic, prev_dic, X_test):
    model_name = "model_"  + str(index)
    new_model = create_keras_model()
    for layer in range(len(new_model.layers)-1):
            new_model.layers[layer].set_weights([prev_dic[model_name][layer][0], prev_dic[model_name][layer][1]])
    
    optim = tf.keras.optimizers.Adam(learning_rate=alpha)
    new_model.compile(optimizer=optim,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    predictions = new_model.predict(X_test, verbose=0)
    multi_predict_dic[model_name] = predictions
    tf.keras.backend.clear_session()
    
    

    
#####################################
def model_prediction_cluster(cluster, client_clusters, multi_precision_dic, multi_recall_dic, multi_f1_score_dic, multi_test_acc_dic, prev_dic, X_test, y_test):
    model_name = "model_"  + str(client_clusters[cluster][0])
    client_name = "client_" + str(client_clusters[cluster][0])
    cluster_name = "cluster_" + str(cluster)
    
    new_model = create_keras_model()
    for layer in range(len(new_model.layers)-1):
            new_model.layers[layer].set_weights([prev_dic[model_name][layer][0], prev_dic[model_name][layer][1]])
    
    optim = tf.keras.optimizers.Adam(learning_rate=alpha)
    new_model.compile(optimizer=optim,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    
    predictions = new_model.predict(X_test, verbose=0)    
    y_pred_fl = np.argmax(predictions, axis = 1)
    
    test_acc_ls = accuracy_score(y_test, y_pred_fl)
    recall_ls = recall_score(y_test, y_pred_fl)
    f1_score_ls = f1_score(y_test, y_pred_fl)
    precision_ls = precision_score(y_test, y_pred_fl)
    
    multi_precision_dic[cluster_name] = precision_ls    
    multi_recall_dic[cluster_name] = recall_ls 
    multi_f1_score_dic[cluster_name] = f1_score_ls 
    multi_test_acc_dic[cluster_name] = test_acc_ls 
    
    
    tf.keras.backend.clear_session()
    
    
    
    
    
#####################################    
def model_prediction_general(index, multi_precision_dic, multi_recall_dic, multi_f1_score_dic, multi_test_acc_dic, prev_dic, X_test, y_test):
    model_name = "model_"  + str(index)
    client_name = "client_" + str(index)
    
    
    new_model = create_keras_model()
    for layer in range(len(new_model.layers)-1):
            new_model.layers[layer].set_weights([prev_dic[model_name][layer][0], prev_dic[model_name][layer][1]])
    
    optim = tf.keras.optimizers.Adam(learning_rate=alpha)
    new_model.compile(optimizer=optim,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    
    predictions = new_model.predict(X_test, verbose=0)    
    y_pred_fl = np.argmax(predictions, axis = 1)
    
    test_acc_ls = accuracy_score(y_test, y_pred_fl)
    recall_ls = recall_score(y_test, y_pred_fl)
    f1_score_ls = f1_score(y_test, y_pred_fl)
    precision_ls = precision_score(y_test, y_pred_fl)
    
    multi_precision_dic[model_name] = precision_ls    
    multi_recall_dic[model_name] = recall_ls 
    multi_f1_score_dic[model_name] = f1_score_ls 
    multi_test_acc_dic[model_name] = test_acc_ls 
    
    
    tf.keras.backend.clear_session()
    
    
    
    
    
# #####################################    
# def model_prediction_general(index, multi_precision_dic, multi_recall_dic, multi_f1_score_dic, multi_test_acc_dic, prev_dic, X_test, y_test):
#     model_name = "model_"  + str(index)
#     client_name = "client_" + str(index)
    
    
#     new_model = create_keras_model()
#     for layer in range(len(new_model.layers)-1):
#             new_model.layers[layer].set_weights([prev_dic[model_name][layer][0], prev_dic[model_name][layer][1]])

#     new_model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    
#     predictions = new_model.predict(X_test, verbose=0)    
#     y_pred_fl = np.argmax(predictions, axis = 1)
    
#     test_acc_ls = accuracy_score(y_test, y_pred_fl)
#     recall_ls = recall_score(y_test, y_pred_fl)
#     f1_score_ls = f1_score(y_test, y_pred_fl)
#     precision_ls = precision_score(y_test, y_pred_fl)
    
#     multi_precision_dic[model_name] = precision_ls    
#     multi_recall_dic[model_name] = recall_ls 
#     multi_f1_score_dic[model_name] = f1_score_ls 
#     multi_test_acc_dic[model_name] = test_acc_ls 
    
    
#     tf.keras.backend.clear_session()
    
    

    
    
    
    
    
    
    
   
    
#########################################    
def encryption_func(index, multi_encrypted_prev_dic, prev_dic, dim_list, public_key):
    model_name = "model_" + str(index)
#     encrypted_prev_dic[model_name] = []
    
    multi_val_temp_list = []
    
    for layer in range(len(dim_list)):
        layer_list = []
        for itr1 in range(2):
            tmp_list = []
            if itr1==0:
                for itr2 in range(dim_list[layer][itr1][0]):
                    w_list = []
                    for itr3 in range(dim_list[layer][itr1][1]):
                        enc_val = public_key.encrypt(int(prev_dic[model_name][layer][itr1][itr2][itr3]*1e15))
                        w_list.append(enc_val)
                    tmp_list.append(w_list)
            else:
                for itr2 in range(dim_list[layer][itr1]):
                    enc_val = public_key.encrypt(int(prev_dic[model_name][layer][itr1][itr2]*1e15))
                    tmp_list.append(enc_val)
            layer_list.append(tmp_list)
        multi_val_temp_list.append(layer_list)
    
    multi_encrypted_prev_dic[model_name] = multi_val_temp_list.copy()
    

    
    
#########################################
def partital_decrypt_func(index, multi_partial_decrypt_dic, encrypted_sum_dic, dim_list, private_key_dic):   
    model_name = "model_" + str(index)
    
    model_priv_key = private_key_dic[model_name]
    
    temp_partial_decrypt_list = []

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
        temp_partial_decrypt_list.append(tmp)



    for layer in range(len(dim_list)):
        for itr1 in range(2):
            if itr1 == 0:
                for itr2 in range(dim_list[layer][itr1][0]):
                    for itr3 in range(dim_list[layer][itr1][1]):                  
                        temp_partial_decrypt_list[layer][itr1][itr2].append(model_priv_key.partialDecrypt(encrypted_sum_dic[model_name][layer][itr1][itr2][itr3]))
            else:
                for itr2 in range(dim_list[layer][itr1]):
                    temp_partial_decrypt_list[layer][itr1].append(model_priv_key.partialDecrypt(encrypted_sum_dic[model_name][layer][itr1][itr2]))
    
    multi_partial_decrypt_dic[model_name] = temp_partial_decrypt_list.copy()

    
    
    
    
    
    
    
    
    
#########################################
class ThresholdPaillier(object):
    def __init__(self,size_of_n):
        #size_of_n = 1024
        pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
        self.p1 = priv.p
        self.q1 = priv.q

        while sympy.isprime(2*self.p1 +1)!= True:
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.p1 = priv.p
        while sympy.isprime(2*self.q1 +1)!= True:
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.q1 = priv.q

        self.p = (2*self.p1) + 1
        self.q = (2*self.q1) + 1
        print(sympy.isprime(self.p),sympy.isprime(self.q),sympy.isprime(self.p1),sympy.isprime(self.q1))
        self.n = self.p * self.q
        self.s = 1
        self.ns = pow(self.n, self.s)
        self.nSPlusOne = pow(self.n,self.s+1)
        self.nPlusOne = self.n + 1
        self.nSquare = self.n*self.n

        self.m = self.p1 * self.q1
        self.nm = self.n*self.m
        self.l = NUM_CLIENTS # Number of shares of private key
        self.w = NUM_CLIENTS # The minimum of decryption servers needed to make a correct decryption.
        self.delta = self.factorial(self.l)
        self.rnd = random.randint(1,1e50)
        self.combineSharesConstant = sympy.mod_inverse((4*self.delta*self.delta)%self.n, self.n)
        self.d = self.m * sympy.mod_inverse(self.m, self.n)

        self.ais = [self.d]
        for i in range(1, self.w):
            self.ais.append(random.randint(0,self.nm-1))

        self.r = random.randint(1,self. p) ## Need to change upper limit from p to one in paper
        while math.gcd(self.r,self.n) != 1:
            self.r = random.randint(0, self.p)
        self.v = (self.r*self.r) % self.nSquare

        self.si = [0] * self.l
        self.viarray = [0] * self.l

        for i in range(self.l):
            self.si[i] = 0
            X = i + 1
            for j in range(self.w):
                self.si[i] += self.ais[j] * pow(X, j)
            self.si[i] = self.si[i] % self.nm
            self.viarray[i] = pow(self.v, self.si[i] * self.delta, self.nSquare)

        self.priv_keys = []
        for i in range(self.l):
            self.priv_keys.append(ThresholdPaillierPrivateKey(self.n, self.l, self.combineSharesConstant, self.w, \
                                            self.v, self.viarray, self.si[i], i+1, self.r, self.delta, self.nSPlusOne))
        self.pub_key = ThresholdPaillierPublicKey(self.n, self.nSPlusOne, self.r, self.ns, self.w,\
                                                 self.delta, self.combineSharesConstant)

    def factorial(self, n):
        fact = 1
        for i in range(1,n+1):
            fact *= i
        return fact

    def computeGCD(self, x, y):
       while(y):
           x, y = y, x % y
       return x



class PartialShare(object):
    def __init__(self, share, server_id):
        self.share = share
        self.server_id =server_id

        
        
class ThresholdPaillierPrivateKey(object):
    def __init__(self,n, l,combineSharesConstant, w, v, viarray, si, server_id, r, delta, nSPlusOne):
        self.n = n
        self.l = l
        self.combineSharesConstant = combineSharesConstant
        self.w = w
        self.v = v
        self.viarray = viarray
        self.si = si
        self.server_id = server_id
        self.r = r
        self.delta = delta
        self.nSPlusOne = nSPlusOne

    def partialDecrypt(self, c):
        return PartialShare(pow(c.c, self.si*2*self.delta, self.nSPlusOne), self.server_id)

    
    
class ThresholdPaillierPublicKey(object):
    def __init__(self,n, nSPlusOne, r, ns, w, delta, combineSharesConstant):
        self.n = n
        self.nSPlusOne = nSPlusOne
        self.r = r
        self.ns =ns
        self.w = w
        self.delta = delta
        self.combineSharesConstant = combineSharesConstant

    def encrypt(self, msg):
        msg = msg % self.nSPlusOne if msg < 0 else msg
        c = (pow(self.n+1, msg, self.nSPlusOne) * pow(self.r, self.ns, self.nSPlusOne)) % self.nSPlusOne
        return EncryptedNumber(c, self.nSPlusOne, self.n)

class EncryptedNumber(object):
    def __init__(self, c, nSPlusOne, n):
        self.c = c
        self.nSPlusOne = nSPlusOne
        self.n = n

    def __mul__(self, cons):
        if cons < 0:
            return EncryptedNumber(pow(sympy.mod_inverse(self.c, self.nSPlusOne), -cons, self.nSPlusOne), self.nSPlusOne, self.n)
        else:
            return EncryptedNumber(pow(self.c, cons, self.nSPlusOne), self.nSPlusOne, self.n)

    def __add__(self, c2):
        return EncryptedNumber((self.c * c2.c) % self.nSPlusOne, self.nSPlusOne, self.n)

    
    
    
    
#########################################    
def combineShares(shrs, w, delta, combineSharesConstant, nSPlusOne, n, ns):
        cprime = 1
        for i in range(w):
            ld = delta
            for iprime in range(w):
                if i != iprime:
                    if shrs[i].server_id != shrs[iprime].server_id:
                        ld = (ld * -shrs[iprime].server_id) // (shrs[i].server_id - shrs[iprime].server_id)
            #print(ld)
            shr = sympy.mod_inverse(shrs[i].share, nSPlusOne) if ld < 0 else shrs[i].share
            ld = -1*ld if ld <1 else ld
            temp = pow(shr, 2 * ld, nSPlusOne)
            cprime = (cprime * temp) % nSPlusOne
        L = (cprime - 1) // n
        result = (L * combineSharesConstant) % n
        return result - ns if result > (ns // 2) else result