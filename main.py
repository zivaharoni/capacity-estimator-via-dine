import os
import sys
import logging
from time import sleep
import argparse
from algorithm import Algorithm
import csv
from utils import preprocess

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
logger = logging.getLogger("logger")

################## Parsing simulation arguments ##################

parser = argparse.ArgumentParser(description='provide arguments for DI estimation')
parser.add_argument('--seed',           type=int,       default=None,     help='random seed for repeatability')
parser.add_argument('--m',              type=int,       default=None,     help='generator dim')
parser.add_argument('--n',              type=int,       default=None,     help='R.V dim')
parser.add_argument('--T',              type=int,       default=None,     help='#of unroll step of RNN')
parser.add_argument('--batch_size',     type=int,       default=None,     help='batch size')
parser.add_argument('--epochs_di',      type=int,       default=None,     help='#of unroll step of RNN')
parser.add_argument('--epochs_enc',     type=int,       default=None,     help='#of unroll step of RNN')
parser.add_argument('--n_steps_mc',     type=int,       default=None,     help='#of unroll step of RNN')
parser.add_argument('--clip_norm',      type=float,     default=None,     help='discount factor of future rewards')
parser.add_argument('--P',              type=float,     default=None,     help='discount factor of future rewards')
parser.add_argument('--C',              type=float,     default=None,     help='discount factor of future rewards')
parser.add_argument('--lr_rate_DI',     type=float,     default=None,     help='training lr')
parser.add_argument('--lr_rate_enc',    type=float,     default=None,     help='training lr')
parser.add_argument('--opt',            type=str,       default=None,     help='opt name')
parser.add_argument('--config_name',    type=str,       default=None,     help='channel name')
parser.add_argument('--name',           type=str,       default=None,     help='simulation name')
parser.add_argument('--verbose',        dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

args = parser.parse_args()


################## Pre-processing simulation configurations ##################
config, logger = preprocess(args)

################## Initiate algorithm ##################

alg = Algorithm(config)


################## Initial evaluation of DI of randomized networks ##################

logger.info('Before training:')

DI = alg.evaluate_encoder(n_steps=config.n_steps_mc)

logger.info("DI  estimation with {} samples: {:2.7f}".format(config.T * config.n_steps_mc * config.batch_size, DI))
logger.info("\n"*3)


################## Initial training of DINE model exclusively ##################

logger.info('training DI only')

history_di = alg.train_mi(n_epochs=config.epochs_di)
DI = alg.evaluate_encoder(n_steps=config.n_steps_mc)

logger.info("DI  estimation with {} samples: {:2.7f}".format(config.T * config.n_steps_mc * config.batch_size, DI))
alg.plot(history_di, 'Training Process of DI only', save=True)
logger.info("\n"*3)

################## Training DINE and NDT models interchangeably ##################

history_enc = alg.train_encoder(n_epochs=config.epochs_enc)

alg.plot(history_enc, 'Training Encoder Process', save=True)



################## Final evaluation of DI for trained DINE and NDT ##################

logger.info('After tranining')
sleep(1)
config.n_steps_mc *= 10
DI = alg.evaluate_encoder(n_steps=config.n_steps_mc)
logger.info("DI  estimation with {} samples: {:2.7f}".format(config.T * config.n_steps_mc * config.batch_size, DI))
logger.info("\n"*3)


################## save results ##################
history_di = [float(h) for h in history_di]
history_enc = [float(h) for h in history_enc]
with open(os.path.join(config.directory,'results.csv'), 'w') as file:
    writer = csv.writer(file)
    writer.writerows(zip(history_di, history_enc))


directory_parent = "{}/results/{}/{}".format(os.path.dirname(sys.argv[0]),
                                                             config.config_name,
                                                             config.name)
f = open(os.path.join(directory_parent,"summary.csv"), "a")
f.write("{:100s},{:010d},{:2.5f}\n".format(config.simulation_name,config.seed, DI))
f.close()

