import logging
logger = logging.getLogger("logger")

class BasicConfig(object):
    ########## general ##########
    name = 'debug'

    ######## randomness #########
    seed = 8564456

    ######## environment ########
    config_name = "awgn"        # channel model
    n = 1                       # dimension of X,Y
    P = 1.0                     # power of X
    N = 1.0                     # power of channel's inner innovation process
    feedback = False            # compute feedback capacity
    C = None                    # capacity of the used configuration (for visualization)
    m = 1                       # dimension of NDT generator


    ######## DINE/NDT models #########
    T = 2                       # unroll of lstm models
    DI_hidden = 512             # hidden units in DINE model
    DI_last_hidden = 256        # hidden units of last layer in DINE model
    DI_dropout = 0.5            # dropout and recurrent dropout in DINE model
    NDT_hidden = 512            # hidden units in NDT model
    NDT_last_hidden = 100       # hidden units of last layer in NDT model
    NDT_dropout = 0.5           # dropout and recurrent dropout in NDT model

    ######## training #########
    opt = "sgd"
    lr_rate_DI = 1.0
    lr_rate_enc = 0.1
    clip_norm = 0.01
    batch_size = 500
    epochs_di = 500
    epochs_enc = 5000
    n_steps_mc = 100

    def show(self):
        attrs = [attr for attr in dir(self) if (not attr.startswith('__') and attr != "show")]
        logger.info('\n'.join("%s: %s" % (item, getattr(self, item)) for item in attrs))


class ConfigAWGN(BasicConfig):
    config_name = "awgn"        # channel model
    n = 1                       # dimension of X,Y
    P = 1.0                     # power of X
    N = 1.0                     # power of channel's inner innovation process
    feedback = False            # compute feedback capacity
    C = 0.3466                  # capacity of the used configuration (for visualization)
    m = 1                       # dimension of NDT generator

class ConfigFF_MA_AGN(BasicConfig):
    config_name = "arma_ff"     # channel model
    channel_alpha = 0.5         # channel parameter
    n = 1                       # dimension of X,Y
    P = 1.0                     # power of X
    N = 1.0                     # power of channel's inner innovation process
    feedback = False            # compute feedback capacity
    C = None                    # capacity of the used configuration (for visualization)
    m = 1                       # dimension of NDT generator

class ConfigFB_MA_AGN(BasicConfig):
    config_name = "arma_fb"     # channel model
    channel_alpha = 0.5         # channel parameter
    n = 1                       # dimension of X,Y
    P = 1.0                     # power of X
    N = 1.0                     # power of channel's inner innovation process
    feedback = True             # compute feedback capacity
    C = None                    # capacity of the used configuration (for visualization)
    m = 1                       # dimension of NDT generator

