#!/usr/bin/env python
__author__ = 'joerg'

""" Phonem Classification on the TIMIT speech corpus with RecNet framework based on Theano """
"""________________________________________________________________________________________"""
"""
"""


######  GLOBAL THEANO CONFIG   #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False" #ast_run
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags

######         IMPORTS          ######
import numpy as np
import sklearn.metrics
from scipy import stats
import time
import recnet


### 1. Step: Create new model
rn = recnet.rnnModel()

### 2. Step: Define parameters
rn.parameter["output_location"] = "log/"
rn.parameter["output_type"    ] = "both"        # console, file, both

rn.parameter["train_data_name"] = "timit_train_xy_mfcc12-26win25-10.klepto"
rn.parameter["valid_data_name"] = "timit_valid_xy_mfcc12-26win25-10.klepto"
rn.parameter["data_location"] = "data_set/"
rn.parameter["batch_size" ] = 10

rn.parameter["net_size"      ] = [     26,      218,        61]
rn.parameter["net_unit_type" ] = ['input', 'GRU_ln', 'softmax']
rn.parameter["net_act_type"  ] = [    '-',   'tanh',       '-']
rn.parameter["net_arch"      ] = [    '-',     'bi',      'ff']

rn.parameter["random_seed"   ] = 211
rn.parameter["epochs"        ] = 20
rn.parameter["optimization"  ] = "adadelta"
rn.parameter["loss_function" ] = "cross_entropy"


### 3. Step: Create model and compile functions
rn.create(['train', 'valid'])

### 4. Step: Train model
rn.pub("Start training")

### 4.1: Create minibatches for validation set
mb_valid_x, mb_valid_y, mb_valid_m = rn.get_mini_batches("valid")

#save measurements
list_ce = []

for i in xrange(rn.prm.optimize["epochs"]):
    time_training_start = time.time()
    time_training_temp = time.time()
    rn.pub("------------------------------------------")
    rn.pub(str(i)+" Epoch, Training run")

    train_error = np.zeros(rn.sample_quantity('train'))

    mb_train_x, mb_train_y, mb_mask = rn.get_mini_batches("train")

    for j in xrange(rn.batch_quantity('train')):

        net_out, train_error[j] = rn.train_fn( mb_train_x[j], mb_train_y[j], mb_mask[j])


        #Insample error
        if ( j%50) == 0 :
            rn.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_training_temp) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_training_temp = time.time()

        #Validation
        if ( (j%500) == 0 or j == rn.batch_quantity('train')-1 ) and j>0:
            rn.pub("###########################################")
            rn.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_error = np.zeros([rn.batch_quantity('valid')])
            corr_error = np.zeros([rn.batch_quantity('valid'),rn.batch_size()])
            ce_error = np.zeros([rn.batch_quantity('valid'),rn.batch_size()])

            for v in np.arange(0,rn.batch_quantity('valid')):
                v_net_out_, v_error[v] = rn.valid_fn(mb_valid_x[v],mb_valid_y[v],mb_valid_m[v])

                for b in np.arange(0,rn.batch_size()):
                    true_out = mb_valid_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]
                    corr_error[v,b] = np.mean(np.argmax(true_out,axis=1)==np.argmax(code_out, axis=1))
                    ce_error[v,b] = sklearn.metrics.log_loss( true_out,code_out)

            list_ce.append(np.mean(v_error))

            array_ce = np.asarray(list_ce[-3:])
            ce_slope, intercept, r_value, p_value, std_err = stats.linregress(range(array_ce.shape[0]),array_ce)

            rn.pub("## cross entropy theano  : " + "{0:.4f}".format(np.mean(v_error)))
            rn.pub("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
            rn.pub("## correct classified    : " + "{0:.4f}".format(np.mean(corr_error)))
            rn.pub("## ce improve      : " + "{0:.6f}".format(ce_slope))
            rn.pub("###########################################")

            rn.dump()
    rn.pub("###########################################")
    rn.pub("Insample Error: " + str(np.mean(train_error)))
    rn.pub("Epoch training duration: "+ str(time.time()-time_training_start) + "sec")

rn.pub("## ||||||||||||||||||||||||||||||||||||||||")


