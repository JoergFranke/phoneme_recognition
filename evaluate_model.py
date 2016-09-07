


######  GLOBAL THEANO CONFIG   #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False" #ast_run
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags


######         IMPORTS          ######
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import sklearn.metrics
from collections import OrderedDict
#import matplotlib.pyplot as plt

from rnnfwk.build_model import rnnModel
from rnnfwk.data_handler import load_minibatches


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

data_name = prm_structure["corpus_name" ] + '_test_' + prm_structure["set_specs" ]
test_mb_set_x,test_mb_set_y,test_mb_set_m = load_minibatches(data_location, data_name, prm_structure["batch_size"])
set_length = test_mb_set_x.__len__()


###### LOAD MODEL
################################################################################
###################### ADD NAME FROM TRAINED MODEL HERE ! ######################
model_name = "outcome/" + "n-*********************.prm"
model_name = "outcome/" + "n-26-200-50-61-bi_d-06-09-2016_v-2.prm"
lstm = rnnModel(None, None, rng, trng, True, model_name, 10)

forward_fn = lstm.get_forward_function()


###### TEST MODEL
ce_error = np.zeros([set_length*prm_structure["batch_size"]])
phn_error = np.zeros([set_length*prm_structure["batch_size"]])

for v in np.arange(0, set_length):
    v_net_out_ = forward_fn(test_mb_set_x[v], test_mb_set_m[v])[0]

    for b in np.arange(0,prm_structure["batch_size"]):
        true_out = test_mb_set_y[v][:, b, :]
        code_out = v_net_out_[:, b, :]

        count = v * prm_structure["batch_size"] + b

        phn_error[count] = np.mean(np.argmax(true_out, axis=1) == np.argmax(code_out, axis=1))
        ce_error[count] = sklearn.metrics.log_loss(true_out, code_out)


print("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
print("## phoneme error rate    : " + "{0:.4f}".format(1 - np.mean(phn_error)))
