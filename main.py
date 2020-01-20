import os
import sys
from time import sleep
import argparse
import tensorflow as tf
from algorithm import Algorithm
from scipy.io import savemat
import csv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


print("Using tensrflow version {}".format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

parser = argparse.ArgumentParser(description='provide arguments for DI estimation')

######## randomness #########
parser.add_argument('--seed',           type=int, default=None,         help='random seed for repeatability')

######## RL #########
parser.add_argument('--m',              type=int, default=1,         help='generator dim')
parser.add_argument('--n',              type=int, default=1,         help='R.V dim')
parser.add_argument('--T',              type=int, default=2,         help='#of unroll step of RNN')
parser.add_argument('--batch',          type=int, default=500,         help='batch size')
parser.add_argument('--epochs_di',      type=int, default=1000,         help='#of unroll step of RNN')
parser.add_argument('--epochs_enc',     type=int, default=10000,         help='#of unroll step of RNN')
parser.add_argument('--n_steps_mc',     type=int, default=100,         help='#of unroll step of RNN')
parser.add_argument('--clipnorm',       type=float, default=1.0,       help='discount factor of future rewards')
parser.add_argument('--P',              type=float, default=1.0,       help='discount factor of future rewards')
parser.add_argument('--C',              type=float, default=0.405,       help='discount factor of future rewards')

######## summary #########
parser.add_argument('--channel',        type=str, default="arma_awgn",         help='simulation name')
parser.add_argument('--name',           type=str, default="debug",         help='simulation name')

parser.add_argument('--feedback',       dest='feedback',                 action='store_true')
parser.set_defaults(feedback=False)


args = parser.parse_args()

m = args.m
n = args.n  # dimention of the R.V. X
T = args.T  # unrol sequential model T time-steps
BATCH = args.batch
CLIPNORM = args.clipnorm
EPOCHS_TRAIN_MI = args.epochs_di
EPOCHS_TRAIN_ENC = args.epochs_enc
N_STEPS_MC = args.n_steps_mc
stateful = True
feedback = args.feedback


NOISE_VAR = float(1.0)
P = args.P  # [0.01, 0.0316, 0.1,  0.316, 1.0, 3.162, 10., 31.6]
CAPACITY = args.C# [0.5*np.log(1+1.0/NOISE_VAR)]#[0.0161, 0.0423, 0.0996, 0.2100, 0.4054, 0.7420, 1.2100, 1.7460]

channel_configs = {"name": args.channel,
                  "noise_var": NOISE_VAR,
                  "alpha": 0.5,
                  "capacity": None}

print("\n\n\n\nTraining for P={:2.5f}\n\n\n\n".format(P))

channel_configs["capacity"] = CAPACITY

directory = os.path.join(os.path.curdir,
                         "results-fb" if feedback else "results-ff",
                         args.channel,
                         args.name,
                         "P={:2.4f}".format(P))

if not os.path.exists(directory):
    os.makedirs(directory)

sys.stdout = open(os.path.join(directory,'log.txt'), 'w')


alg = Algorithm(n=n,
                P=P,
                channel_configs=channel_configs,
                m=m,
                T=T,
                batch=BATCH,
                epochs_train_di=EPOCHS_TRAIN_MI,
                epochs_train_enc=EPOCHS_TRAIN_ENC,
                clipnorm=CLIPNORM,
                n_steps_mc=N_STEPS_MC,
                stateful=stateful,
                directory=directory,
                feedback=feedback)


directory_parent = os.path.join(os.path.curdir,
                         "results-fb" if feedback else "results-ff",
                         args.channel,
                         args.name)
# In[9]:

print('Before training:')
sleep(1)
DI = alg.evaluate_encoder(n_steps=N_STEPS_MC)
print("DI  estimation with {} samples: {:2.7f}".format(T * N_STEPS_MC * BATCH, DI))
# alg.hist_X(tf.reshape(alg.rand_enc_samples(N_STEPS_MC), [-1]), 'encoder dist')


print('training DI only')
sleep(1)
history_di = alg.train_mi(n_epochs=EPOCHS_TRAIN_MI)
DI = alg.evaluate_encoder(n_steps=N_STEPS_MC)
print("DI  estimation with {} samples: {:2.7f}".format(T * N_STEPS_MC * BATCH, DI))
alg.plot(history_di, 'Training Process of DI only', save=True)


history_enc = alg.train_encoder(n_epochs=EPOCHS_TRAIN_ENC)
alg.plot(history_enc, 'Training Encoder Process', save=True)


print('After tranining')
sleep(1)
N_STEPS_MC *= 100
DI = alg.evaluate_encoder(n_steps=N_STEPS_MC)
print("DI  estimation with {} samples: {:2.7f}".format(T * N_STEPS_MC * BATCH, DI))
# alg.hist_X(tf.reshape(alg.rand_enc_samples(N_STEPS_MC), [-1]), 'encoder dist', save=True)
# savemat(os.path.join(directory_parent,"results.mat"), {"history_di": history_di,
#                                                         "history_enc": history_enc,
#                                                         "DI": DI})

history_di = [float(h) for h in history_di]
history_enc = [float(h) for h in history_enc]
with open(os.path.join(directory,'results.csv'), 'w') as file:
    writer = csv.writer(file)
    writer.writerows(zip(history_di, history_enc))

f = open(os.path.join(directory_parent,"summary.csv"), "a")
f.write("{:2.5f}, {:2.5f}\n".format(P, DI))
f.close()
# alg.save()

