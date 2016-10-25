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
from collections import OrderedDict

from recnet.build_model import rnnModel


# ### 1. Step: Define parameters
parameter = OrderedDict()
parameter["output_location"] = "log/"
parameter["output_type"    ] = "both"        # console, file, both

parameter["train_data_name"] = "timit_train_xy_mfcc12-26win25-10.klepto"
parameter["valid_data_name"] = "timit_valid_xy_mfcc12-26win25-10.klepto"
parameter["data_location"] = "data_set/"
parameter["batch_size" ] = 10

parameter["net_size"      ] = [     26,      218,        61]
parameter["net_unit_type" ] = ['input', 'GRU_ln', 'softmax']
parameter["net_act_type"  ] = [    '-',   'tanh',       '-']
parameter["net_arch"      ] = [    '-',     'bi',      'ff']

parameter["random_seed"   ] = 211
parameter["epochs"        ] = 20
parameter["optimization"  ] = "adadelta"  # sgd, nm_rmsprop, rmsprop, nesterov_momentum, adadelta
parameter["loss_function" ] = "cross_entropy"


### 2. Step: Create new model
model = rnnModel(parameter)


### 3. Step: Build model functions
train_fn    = model.get_training_function()
valid_fn    = model.get_validation_function()


### 4. Step: Train model
model.pub("Start training")
valid_mb_set_x, valid_mb_set_y, valid_mb_set_m = model.get_mini_batches("valid")

#save measurements
list_ce = []

for i in xrange(model.prm.optimize["epochs"]):
    time_training_start = time.time()
    time_training_temp = time.time()
    model.pub("------------------------------------------")
    model.pub(str(i)+" Epoch, Training run")

    train_error = np.zeros(model.prm.data["train_set_len" ])

    mb_train_x, mb_train_y, mb_mask = model.get_mini_batches("train")

    for j in xrange(model.prm.data["train_batch_quantity"]):

        net_out, train_error[j] = train_fn( mb_train_x[j],
                                            mb_train_y[j],
                                            mb_mask[j]
                                            )


        #Insample error
        if ( j%50) == 0 :
            model.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_training_temp) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_training_temp = time.time()

        #Validation
        if ( (j%500) == 0 or j == model.prm.data["train_batch_quantity" ]-1 ) and j>0:
            model.pub("###########################################")
            model.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_error = np.zeros([model.prm.data["valid_batch_quantity"]])
            corr_error = np.zeros([model.prm.data["valid_batch_quantity"],model.prm.data["batch_size"]])
            ce_error = np.zeros([model.prm.data["valid_batch_quantity"],model.prm.data["batch_size"]])

            for v in np.arange(0,model.prm.data["valid_batch_quantity"]):
                v_net_out_, v_error[v] = valid_fn(valid_mb_set_x[v],valid_mb_set_y[v],valid_mb_set_m[v])

                for b in np.arange(0,model.prm.data["batch_size"]):
                    true_out = valid_mb_set_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]
                    corr_error[v,b] = np.mean(np.argmax(true_out,axis=1)==np.argmax(code_out, axis=1))
                    ce_error[v,b] = sklearn.metrics.log_loss( true_out,code_out)

            list_ce.append(np.mean(v_error))

            array_ce = np.asarray(list_ce[-3:])
            ce_slope, intercept, r_value, p_value, std_err = stats.linregress(range(array_ce.shape[0]),array_ce)

            model.pub("## cross entropy theano  : " + "{0:.4f}".format(np.mean(v_error)))
            model.pub("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
            model.pub("## correct classified    : " + "{0:.4f}".format(np.mean(corr_error)))
            model.pub("## ce improve      : " + "{0:.6f}".format(ce_slope))
            model.pub("###########################################")

            model.dump()
    model.pub("###########################################")
    model.pub("Insample Error: " + str(np.mean(train_error)))
    model.pub("Epoch training duration: "+ str(time.time()-time_training_start) + "sec")

model.pub("## ||||||||||||||||||||||||||||||||||||||||")


