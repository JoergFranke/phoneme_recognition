#!/usr/bin/env python
__author__ = 'joerg'

######  GLOBAL THEANO CONFIG   #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False" #ast_run
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags


######         IMPORTS          ######
import numpy as np
import sklearn.metrics
from collections import OrderedDict
import matplotlib.pyplot as plt
import recnet

### 1. Step: Create new model
rn = recnet.rnnModel()

### 2. Step: Define parameters
rn.parameter["load_model"] = True
rn.parameter["model_location"] = "model_save/"
###############################################################################
########################### ADD NAME FROM TRAINED MODEL HERE ! #################
rn.parameter["model_name"] = "GRU_ln-softmax_26-218-61_bi_d-18-11-2016_v-1.prm" #**********************************.prm"
rn.parameter["batch_size" ] = 5
rn.parameter["data_location"] = "data_set/"
rn.parameter["test_data_name"] = "timit_test_xy_mfcc12-26win25-10.klepto"

### 3. Step: Create model and compile functions
rn.create(['forward'])

### 4. Step: Get mini batches from your test data set
test_mb_set_x, test_mb_set_y, test_mb_set_m = rn.get_mini_batches("test")

### 5. Step: Test model
ce_error = np.zeros([rn.sample_quantity('test')])
phn_error = np.zeros([rn.sample_quantity('test')])

for v in np.arange(0, rn.batch_quantity('test')):
    v_net_out_ = rn.forward_fn(test_mb_set_x[v], test_mb_set_m[v])

    for b in np.arange(0,rn.batch_size()):
        true_out = test_mb_set_y[v][:, b, :]
        code_out = v_net_out_[:, b, :]

        count = v * rn.batch_size() + b

        phn_error[count] = np.mean(np.argmax(true_out, axis=1) == np.argmax(code_out, axis=1))
        ce_error[count] = sklearn.metrics.log_loss(true_out, code_out)

print("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
print("## phoneme error rate    : " + "{0:.4f}".format(1 - np.mean(phn_error)))


### 5. Step: Make a sample plot
sample = 1
batch = 1

sample_x = np.asarray(test_mb_set_x[sample][:,batch,:])
sample_y = np.asarray(test_mb_set_y[sample][:,batch,:])
mask = np.asarray(test_mb_set_m[sample][:,batch,:])
sample_x = sample_x[0:int(mask[:,0].sum()),:]
sample_y = sample_y[0:int(mask[:,0].sum()),:]

net_out = rn.forward_fn(test_mb_set_x[sample], test_mb_set_m[sample])
net_out = np.asarray(net_out[:,batch,:])
net_out = net_out[0:int(mask[:,0].sum()),:]
plt.clf()

plt.subplot(3,1,1)
plt.imshow(sample_x.transpose())
plt.xlim([0,int(mask[:,0].sum())])
#plt.xlabel('time steps')
plt.ylabel('input features')


plt.subplot(3,1,2)
plt.plot(sample_y)
plt.ylim([0,1.2])
plt.xlim([0,int(mask[:,0].sum())])
#plt.xlabel('time steps')
plt.ylabel('correct labels')

plt.subplot(3,1,3)
plt.plot(net_out)
plt.ylim([0,1.2])
plt.xlim([0,int(mask[:,0].sum())])
plt.xlabel('time steps')
plt.ylabel('network output')


plt.show()
