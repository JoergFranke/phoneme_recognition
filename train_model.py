#!/usr/bin/env python
__author__ = 'joerg'

""" Phonem Classification on the TIMIT speech corpus with Theano based DBLSTM Network """
"""___________________________________________________________________________________"""
"""
"""


######  GLOBAL THEANO CONFIG   #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False" #ast_run
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags

######      THEANO CONFIG      #######
import theano
#theano.config.device='gpu0'
#theano.config.floatX = 'float32'
#theano.config.mode = 'FAST_RUN'
#theano.config.optimizer = 'fast_run'
#theano.config.allow_gc = False

#theano.config.lib.cnmem =1
theano.config.scan.allow_gc = False
#theano.config.optimizer_excluding ='low_memory'
#theano.config.scan.allow_output_prealloc = True
#theano.config.exception_verbosity='high'

######         IMPORTS          ######
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
import sklearn.metrics
from scipy import stats
import time
from collections import OrderedDict



from rnnfwk.build_model import rnnModel
from rnnfwk.data_handler import load_minibatches


###### GLOBAL TIMER
time_0 = time.time()

########## RANDOM STREAMS
prm_optimization = OrderedDict()
prm_optimization["seed"] = 211
rng = np.random.RandomState(prm_optimization["seed"])
trng = RandomStreams(prm_optimization["seed"] )



###### DATA IN
print("# Load data")
prm_structure = OrderedDict()
prm_structure["batch_size"    ] = 10
prm_structure["set_specs" ] = "xy_mfcc12-26win25-10"
prm_structure["corpus_name" ] = "timit"

data_location = "data_set/"

data_name = prm_structure["corpus_name" ] + '_train_' + prm_structure["set_specs" ]
train_mb_set_x,train_mb_set_y,train_mb_set_m = load_minibatches(data_location, data_name, prm_structure["batch_size"])

data_name = prm_structure["corpus_name" ] + '_valid_' + prm_structure["set_specs" ]
valid_mb_set_x,valid_mb_set_y,valid_mb_set_m = load_minibatches(data_location, data_name, prm_structure["batch_size"])



input_size = train_mb_set_x[0].shape[2]
output_size = train_mb_set_y[0].shape[2]


print "# Loading duration: ",time.time()-time_0 ," sec"


#### Hyper parameter

prm_structure["net_size"      ] = [input_size,200,50, output_size]
prm_structure["hidden_layer"  ] = prm_structure["net_size"].__len__() - 2
prm_structure["bi_directional"] = True
prm_structure["identity_func" ] = False
prm_structure["train_set_len" ] = train_mb_set_x.__len__()
prm_structure["valid_set_len" ] = valid_mb_set_x.__len__()

if "log" not in os.listdir(os.getcwd()):
    os.mkdir("log")
prm_structure["output_location"] = "log/"
prm_structure["output_type"    ] = "both"        # console, file, both


prm_optimization["epochs"        ] = 20
prm_optimization["learn_rate"    ] = 0.0001
prm_optimization["lr_decline"    ] = 0.95
prm_optimization["momentum"      ] = 0.9
prm_optimization["decay_rate"    ] = 0.9
prm_optimization["use_dropout"   ] = False       # False, True
prm_optimization["dropout_level" ] = 0.5
prm_optimization["regularization"] = False       # False, L2, ( L1 )
prm_optimization["reg_factor"    ] = 0.01
prm_optimization["optimization"  ] = "adadelta"  # sgd, nm_rmsprop, rmsprop, nesterov_momentum, adadelta
prm_optimization["noisy_input"   ] = False       # False, True
prm_optimization["noise_level"   ] = 0.6
prm_optimization["loss_function" ] = "cross_entropy" # w2_cross_entropy, cross_entropy
prm_optimization["bound_weight"  ] = 3       # False, Integer (2,12)


###### Build model

lstm = rnnModel(prm_structure, prm_optimization, rng, trng)

lstm.print_model_params()
lstm.pub("# Build model")


time_1 = time.time()
lstm.pub("Model build time"+ str(time_1-time_0) + "sec")

train_fn    = lstm.get_training_function()
valid_fn    = lstm.get_validation_function()
forward_fn  = lstm.get_forward_function()

###### START TRAINING
lstm.pub("Start training")

batch_order = np.arange(0,prm_structure["train_set_len"])

#save measurements
list_ce = []




for i in xrange(prm_optimization["epochs"]):
    time_training_start = time.time()
    time_training_temp = time.time()
    lstm.pub("------------------------------------------")
    lstm.pub(str(i)+" Epoch, Training run")


    train_error = np.zeros(prm_structure["train_set_len" ])
    batch_permut = rng.permutation(batch_order)

    for j in batch_order:

        train_error[j], net_out = train_fn( train_mb_set_x[batch_permut[j]],
                                            train_mb_set_y[batch_permut[j]],
                                            train_mb_set_m[batch_permut[j]]
                                            )


        #Insample error
        if ( j%50) == 0 :
            lstm.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_training_temp) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_training_temp = time.time()

        #Validation
        if ( (j%500) == 0 or j == prm_structure["train_set_len" ]-1 ) and j>0:
            lstm.pub("###########################################")
            lstm.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_error = np.zeros([prm_structure["valid_set_len"]])
            corr_error = np.zeros([prm_structure["valid_set_len"],prm_structure["batch_size"]])
            ce_error = np.zeros([prm_structure["valid_set_len"],prm_structure["batch_size"]])

            for v in np.arange(0,prm_structure["valid_set_len"]):
                v_net_out_, v_error[v] = valid_fn(valid_mb_set_x[v],valid_mb_set_y[v],valid_mb_set_m[v])

                for b in np.arange(0,prm_structure["batch_size"]):
                    true_out = valid_mb_set_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]
                    corr_error[v,b] = np.mean(np.argmax(true_out,axis=1)==np.argmax(code_out, axis=1))
                    ce_error[v,b] = sklearn.metrics.log_loss( true_out,code_out)

            list_ce.append(np.mean(v_error))

            array_ce = np.asarray(list_ce[-3:])
            ce_slope, intercept, r_value, p_value, std_err = stats.linregress(range(array_ce.shape[0]),array_ce)

            lstm.pub("## cross entropy theano  : " + "{0:.4f}".format(np.mean(v_error)))
            lstm.pub("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
            lstm.pub("## correct classified    : " + "{0:.4f}".format(np.mean(corr_error)))
            lstm.pub("## ce improve      : " + "{0:.6f}".format(ce_slope))
            lstm.pub("###########################################")

            lstm.dump()
    lstm.pub("###########################################")
    lstm.pub("Insample Error: " + str(np.mean(train_error)))
    lstm.pub("Epoch training duration: "+ str(time.time()-time_training_start) + "sec")

#Finale Test
lstm.pub("## ||||||||||||||||||||||||||||||||||||||||")


