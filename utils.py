import tensorflow as tf
import sys
import os
import logging
import time
import shutil
from configs import ConfigAWGN, ConfigFF_MA_AGN, ConfigFB_MA_AGN

logger = logging.getLogger("logger")


def define_configs(args):
    if args.config_name == "awgn":
        config = ConfigAWGN()
    elif args.config_name == "arma_ff":
        config = ConfigFF_MA_AGN()
    elif args.config_name == "arma_fb":
        config = ConfigFB_MA_AGN()
    else:
        raise ValueError("Invalid choice of configuration")

    config = read_flags(config, args)

    seed_tmp = time.time()
    config.seed = int((seed_tmp - int(seed_tmp))*1e6) if args.seed is None else args.seed
    print(config.seed)

    simulation_name = get_simulation_name(config)
    config.simulation_name = simulation_name
    config.directory = directory = "{}/results/{}/{}/{}/{}".format(os.path.dirname(sys.argv[0]),
                                                             config.config_name,
                                                             config.name,
                                                             simulation_name,
                                                             config.seed)

    create_exp_dir(directory, scripts_to_save=['algorithm.py',
                                               'configs.py',
                                               'main.py',
                                               'rnn_modified.py',
                                               'utils.py'])

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

    return config


def read_flags(config, args):
    # assign flags into config
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            setattr(config, key, val)

    return config


def get_simulation_name(args):
    waiver = ['seed', 'verbose', 'config_name', 'name']
    name = []
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if key == name:
            continue
        if val is not None and key not in waiver:
            name.append(key + "-" + str(val).replace(",","-").replace(" ", "").replace("[", "").replace("]", ""))
    return "{}".format("_".join(name))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]),script), dst_file)


def define_logger(args, directory):
    logFormatter = logging.Formatter("%(message)s")
    logger = logging.getLogger("logger")

    logger.setLevel(logging.INFO)


    fileHandler = logging.FileHandler("{0}/logger.log".format(directory))

    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    if args.verbose:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    return logger


def preprocess(args):
    ###################################### general configs ######################################

    config = define_configs(args)
    logger = define_logger(args, config.directory)

    logger.info("\n"*10)
    logger.info("cmd line: python " + " ".join(sys.argv) + "\n"*2)
    logger.info("Simulation configurations:\n" + "-"*30)
    config.show()
    logger.info("\n" * 5)
    return config, logger
