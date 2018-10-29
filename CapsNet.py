# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 23:25:28 2018

@author: Meet
"""
import tensorflow as tf
import numpy as np
from numpy import zeros

class CapsNet:
    def __init__(self, pCapsDims, dgtCapsDims, priCapsModules, priCapsDpth, routing_iter, b_size, eps):
        self.capsule_primary_dim = pCapsDims            # 8 dimension for initial capsule = primaryCapsules
        self.capsule_digit_dim = dgtCapsDims            # 16 dimension for digit capsules
        self.primaryCapsModules = priCapsModules        # total 32 modules for primary caps layer having (6x6x8) dimension    
        # Note: primary Capsule Depth is equal to number of class to be predicted.
        self.primaryCapsDepth = priCapsDpth             # this depth is the primary capsule which is 10 which is equal to no of classes
        self.routing_iteration = routing_iter           # no of routing iteration during feed forward operation = 3 or 1
        self.batch_size = b_size                        # generally we dont required batch_size while defining model but here to facilitate matmul operations we need to tile few tensors before matmul and for tiling operation we need batch_size
        self.epsilon = eps                              # epsilon to be used in denomination in squashing activation to prevent division by zero
        self.recon_loss_factor = 0.0005 * 100           # reconstruction loss factor to reduce the importance of reconstruction loss w.r.t. marginloss
        # in paper they used 0.0005 as reconstruction loss factor but to better visualize the reconstructed input it is multiplied with 100
        
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input_x')
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name='label_y')
        
        # x input shape = [batch_size x 28 x 28 x 1]
        self.conv1 = tf.layers.conv2d(self.x, filters=256, kernel_size=(9,9), strides=1, activation=tf.nn.relu, padding='valid', 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.initializers.zeros)
        # conv1 shape = [batch_size x 20 x 20 x 256]
        
        self.primaryCaps = tf.layers.conv2d(self.conv1, filters=self.capsule_primary_dim * self.primaryCapsModules, kernel_size=(9,9), strides=2, activation=tf.nn.relu, padding='valid',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.initializers.zeros)
        # primaryCaps shape = [batch_size x 6 x 6 x 256]
        
        self.primaryCaps = tf.reshape(self.primaryCaps, shape=[-1, self.primaryCapsModules * 6 * 6, self.capsule_primary_dim])
        # primaryCaps shape = [batch_size x 1152 x 8]
        
        # Add a axis just to facilitate tf.matmul with 16 dimension weights
        self.primaryCaps = tf.expand_dims(self.primaryCaps, axis=2)               # [batch_size x 1152 x 1 x 8]        
        self.primaryCaps = tf.expand_dims(self.primaryCaps, axis=1)               # [batch_size x 1 x 1152 x 1 x 8]
        
        # stack 10 identical copies of primaryCaps just because we can do tf.matmul between 8 dims vectors and 16 dims vectors
        self.primaryCaps = tf.tile(self.primaryCaps, multiples=[1, self.primaryCapsDepth, 1, 1, 1])    # [batch_size x 10 x 1152 x 1 x 8]
        # NOTE ON TF.TILE OPERATION : The tf.tile operation will just stack identical copies as per given location. So all the tensors are having identical values.
        # IMP NOTE on tf.tile       : While doing the backprop with tf.tile operation the backprop will be done to the original primaryCaps tensor which is [b x 1 x 1152 x 1 x 8] instead of [b x 10 x 1152 x 1 x 8] tiled tensor as all the tensors in tiled tensor are identical.
        
        self.u = self.squash(self.primaryCaps, axis=4)                                 # [batch_size x 10 x 1152 x 1 x 8]
        
        # now we are converting each capsules from [1 x 8] dims to [1 x 16] dims by doing matmul with [8 x 16] weights for each capsule
        self.w_caps = tf.Variable(initial_value=tf.random_normal(stddev=0.1, shape=[1, self.primaryCapsDepth, 1152, self.capsule_primary_dim, self.capsule_digit_dim]))
        # tile w_caps just to facilitate matmul
        self.w_caps = tf.tile(self.w_caps, multiples=[self.batch_size, 1, 1, 1, 1])    # [batch_size x 10 x 1152 x 8 x 16]
        
        """
        [batch_size x 10 x 1152 x 1 x 8 ] - u
        [batch_size x 10 x 1152 x 8 x 16] - w_caps
        
        Matrix Multiplication between u and w_caps will give us following dims
        [batch_size x 10 x 1152 x 1 x 16] - u_hat
        """
        
        # matmul to increase the capsule dimension from 8 to 16 so that output digitCaps will have 16 dims capsule dims
        self.u_hat = tf.matmul(self.u, self.w_caps)                            # [batch_size x 10 x 1152 x 1 x 16]
        
        # b_ij are the scalar values which will be multiplied with the u_hat. There will be single b_ij value for each of the capsules.
        # There are total 1152 x 10 capsules in u_hat in each batch. So there will be 1152 x 10 such scalars in a batch
        # NOTE: THESE B_IJ ARE TF.CONSTANT WHICH MEANS THEY WILL NOT BE TRAINED DURING OPTIMIZATION PROCESS. 
        #       BUT THESE ARE UPDATED EXPLICITILY DURING TRAINING WHEN BELOW FOR LOOP WILL BE EXECUTED TO OBTAIN V_IJ
        #       THE UPDATE IN B_IJ TAKES PLACE DUE TO "ROUTING BY AGREMENT ALGORITHM" PROPOSED BY PAPER AUTHORS.
        #       DURING FORWARD PASS B_IJ WILL BE UPDATED.
        
        # For each batch b_ij are initialized at zero at the start of training.
        self.b_ij = tf.constant(zeros(shape=[1, self.primaryCapsDepth, 1152], dtype=np.float32), dtype=tf.float32)   
        # [1 x 10 x 1152]
        # tile b_ij just to facilitate scalar multiplication
        b_ij_tile = tf.tile(self.b_ij, multiples=[self.batch_size, 1, 1])
        # [batch_size x 10 x 1152]
    
        for i in range(self.routing_iteration):
            # expand dims to facilitate scalar multiplication
            b_ij_temp = tf.expand_dims(b_ij_tile, axis=3)                # [batch_size x 10 x 1152 x 1]
         
            # Perform softmax across Capsule depth just to make the weights sum to 1 and positive.
            # That will make weights connecting to single capsule in next layer and all the capsules of current layer sum to 1.
            self.c_ij = tf.nn.softmax(b_ij_temp, axis=1)                 # [batch_size x 10 x 1152 x 1] 
            
            """
            # How to do SCALAR multiplication between u_hat and c_ij? and How to obtain s_j having shape = [batch_size x 10 x 16]?
            # each capsule will be multiplied by unique scalar c_ij
            
            # [batch_size x 10 x 1152 x 1 x 16] - u_hat
            # [batch_size x 10 x 1152 x 1] - c_ij
            
            # [batch_size x 10 x 1152 x 16] - u_hat_squeeze
            # [batch_size x 10 x 1152 x 1] - c_ij            
            
            # scalar multiplication output = [batch_size x 10 x 1152 x 16]
            # output_reduce_sum across axis 2 = [batch_size x 10 x 16]
            """
            
            u_hat_temp = tf.squeeze(self.u_hat)                             # [batch_size x 10 x 1152 x 16]
            
            self.s_j = tf.reduce_sum(u_hat_temp * self.c_ij, axis=2)        # [batch_size x 10 x 1152 x 16] ==> [batch_size x 10 x 16]
                        
            # Squash the output against capsule dimension
            self.v_j = self.squash(self.s_j, axis=2)                        # [batch_size x 10 x 16]
            
            
            """
            # if both u_hat (output of primaryCaps) and v_j (output of digitCaps) agree on identifying the presence of low level (such as eyes, nose) and high level features (such as face) respectively,
            # then their scalar multiplication will be higher and that is what we want to have when we get to know that there are eyes as well as face (decided by several presence of low level features such as eyes, nose, etc)
            # then it has high chances that it is a face.
            # This is what routing by agreement algorithm means
            # and the scalar multiplication value will be added to the b_ij which eventually responsible for the connection between that low level and that high level feature.
            
            # How to do SCALAR multiplication between u_hat_temp and v_j to obtain updated b_ij? and How to obtain shape of b_j = [batch_size x 10 x 1152]?
            # [batch_size x 10 x 16] - v_j
            # [batch_size x 10 x 1152 x 16] - u_hat_temp
            
            # [batch_size x 10 x 1    x 16] - v_j_expand_dims
            # [batch_size x 10 x 1152 x 16] - u_hat_temp
            
            # scalar multiplication output = [batch_size x 10 x 1152 x 16]
            # REDUCED SUM ACROSS AXIS 3 and 0 = [10 x 1152] 
            # Expand dims axis 0 = [1 x 10 x 1152]      and add it to b_ij
            """
            if i < self.routing_iteration - 1:
                v_j_temp = tf.expand_dims(self.v_j, axis=2)                     # [batch_size x 10 x 1 x 16]
                self.b_ij += tf.expand_dims(tf.reduce_sum(tf.reduce_sum(u_hat_temp * v_j_temp, axis=3), axis=0), axis=0)       # [batch_size x 10 X 1152 x 16] ==> [batch_size x 10 X 1152] ==> [10 x 1152] ==> [1 x 10 x 1152]
        
        # now the correct labeled capsule will be extracted and used as an input to fully connected layers 
        self.v_masked = tf.multiply(self.v_j, tf.reshape(self.y, shape=(-1,self.primaryCapsDepth,1)))          # [batch_size x 10 x 16]
        self.v_masked = tf.reduce_sum(self.v_masked, axis=1)                                # [batch_size x 16]
        
        # FC layers just to visualize what the 16 dims vector has embedded inside them. 
        # So we will train the FC layer to reconstruct the input image 
        # so that after training we can see what changes in 16 dims vector makes what changes in the reconstructed image.
        # this will give us detailed insights about the 16 dims capsules.
        self.layer1 = tf.layers.dense(self.v_masked, 512, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer())
        self.layer2 = tf.layers.dense(self.layer1, 1024, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer())
        self.layer3 = tf.layers.dense(self.layer2, 784, activation=tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer())
        # last layer activation = tf.nn.sigmoid just to have the intensities between [0, 1]
        
    def imageReconstruction(self):
        self.inputReconstructed = tf.reshape(self.layer3, shape=(self.batch_size, 28, 28, 1))
        return self.inputReconstructed
    
    def marginLoss(self, m_plus, m_minus, lambda_):
        v_j_bar = tf.sqrt(tf.reduce_sum(tf.square(self.v_j), axis=2))       # [batch_size x 10]
        # self.y shape [batch_size x 10]
        self.margin_loss = self.y * (tf.maximum(0.0, m_plus - v_j_bar)**2) + lambda_ * (1 - self.y) * (tf.maximum(0.0, v_j_bar - m_minus)**2)
        self.margin_loss = tf.reduce_mean(self.margin_loss)
        return self.margin_loss
        
    def reconstructionLoss(self):
        self.reconstructionLoss = self.recon_loss_factor * tf.losses.mean_squared_error(tf.reshape(self.x, (-1, 784)), self.layer3)
        return self.reconstructionLoss
    
    def loss(self, m_p, m_n, labd_):
        return self.marginLoss(m_p, m_n, labd_) + self.reconstructionLoss()
    
    def accuracy(self):
        self.pred_y = tf.sqrt(tf.reduce_sum(tf.square(self.v_j), axis=2))
        self.correct_pred = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.pred_y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float64))
        return self.accuracy
    
    def squash(self, a, axis):
        # a : vector to be squashed
        # axis : capsule dimension axis across which squashing needs to be done
        square = tf.reduce_sum(tf.square(a), axis=axis, keepdims=True)
        scalarMultiplier = square / (1 + square) / (tf.sqrt(square) + self.epsilon)
        return tf.multiply(scalarMultiplier, a)
