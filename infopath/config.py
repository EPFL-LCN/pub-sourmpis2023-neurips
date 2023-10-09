import numpy as np
import torch
import time
from types import SimpleNamespace
import os
import json
import pickle
from optparse import OptionParser
import copy


def load_opt_old(log_path):
    default = get_default_opt()
    try:
        opt = pickle.load(open(os.path.join(log_path, "opt.pickle"), "rb"))
    except:
        opt = vars(default)
    opt_new = vars(default)
    opt = vars(opt)
    for i in opt_new:
        if i not in opt.keys():
            opt[i] = opt_new[i]

    parser = OptionParser()
    for i in opt:
        parser.add_option("--" + i, default=opt[i])
    opt, _ = parser.parse_args()
    return opt


def compare_opt(log_path1, log_path2):
    opt1 = load_training_opt(log_path1)
    opt2 = load_training_opt(log_path2)
    for key in vars(opt1).keys():
        if key not in vars(opt2).keys():
            # print(key, "this is doesn't exist")
            continue
        if key == "log_path":
            continue
        if vars(opt1)[key] != vars(opt2)[key]:
            print(key, vars(opt1)[key], vars(opt2)[key])


def load_training_opt(log_path):
    default = get_default_opt()
    with open(os.path.join(log_path, "opt.json"), "rb") as f:
        opt = json.load(f)
        opt = SimpleNamespace(**opt)
        opt = scale_function(opt)
        opt.device = torch.device(opt.device)
    opt_new = vars(default)
    opt = vars(opt)
    for i in opt_new:
        if i not in opt.keys():
            opt[i] = opt_new[i]
    opt = SimpleNamespace(**opt)
    return opt


def save_opt(log_path, opt):
    # result_path_pickle = os.path.join(log_path, "opt.pickle")
    result_path_json = os.path.join(log_path, "opt.json")
    # with open(result_path_pickle, "wb") as f:
    #     pickle.dump(opt, f)
    with open(result_path_json, "w") as f:
        optsave = copy.deepcopy(opt)
        optsave = vars(optsave)
        optsave["device"] = str(optsave["device"])
        optsave["scale_fun"] = str(optsave["scale_fun"]).split(" ")[1]
        json.dump(optsave, f)


def scale_function(opt):
    if opt.scale_fun == "sigmoid":
        fun = sigmoid
    elif opt.scale_fun == "linear":
        fun = linear
    elif opt.scale_fun == "log":
        fun = log
    else:
        raise NotImplementedError(
            "Scale Function: {} not implemented".format(opt.scale_fun)
        )

    opt.scale_fun = fun

    return opt


def sigmoid(x, opt):
    mean = int(opt.n_stim_amp / 2)
    if x != 0:
        return np.exp(x - mean) / (1 + np.exp(x - mean))
    else:
        return 0


def linear(x):
    return x


def log(x):
    return np.log(1 + x)


def get_default_opt():
    default = {
        # directory of the dataset
        "datapath": "./datasets",
        # Device to use {either cpu or cuda:0}
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        # how often to log
        "log_every_n_steps": 50,
        # batch size
        "batch_size": 50,
        # learning rate
        "lr": 0.0003,
        # weight decay
        "w_decay": 0.01,
        # l1 decay
        "l1_decay": 0.01,
        #  how many neurons
        "n_units": 300,
        # what kind of rnn
        "rnn_type": "rsnn",
        # Name of task (for readable logs)
        "task": "brain_encoding",
        # Number of training steps
        "n_steps": 100000,
        # start in seconds relative to onset
        "start": -0.1,
        # stop in seconds relative to onset
        "stop": 0.5,
        # possible stimuli amplitudes
        "n_stim_amp": 4,
        # Onset of trial in seconds
        "trial_onset": 1.0,
        # duration of stimulus in seconds
        "stim_duration": 0.01,
        # Frequency in Hz of the input neurons
        "input_f0": 2.0,
        # Timestep of simulation in ms
        "dt": 1.0,
        # Number of inputs to the RNNs
        "n_rnn_in": 128,
        # function to scale differnt stim amplitudes, it can be {'sigmoid', 'linear', 'log'}
        "scale_fun": "log",
        # valance of the stimulus, multiplier of scale_fun
        "stim_valance": 5,
        # how impact has the regularization from firing rate distribution
        "coeff_firing_rate_distro_reg": 0.0,
        # how impact has the main loss to the total loss
        "coeff_loss": 1.0,
        # if the spike inputs are seeded else random
        "input_seed": 0,
        # how strong is the membrane potential noise, in each area
        "noise_level_list": [0.16, 0.16, 0.16],
        # timeconstant of membrane potential of areas in ms
        "tau_list": [10.0, 10.0, 10.0],
        # timeconstant of synaptic condactunce of areas in ms
        "tau_syn_list": [2.0, 2.0, 2.0],
        # timeconstant of adaptative threshold in ms
        "tau_adaptation": 144.0,
        # name of areas
        "areas": ["wS1", "mPFC", "tjM1"],
        # delay from input to rnn in seconds
        "thalamic_delay": 0.005,
        # synaptic delay for intra area
        "n_delay": 5,
        # synaptic delay for inter area
        "inter_delay": 5,
        # more readable way to set stim
        "stim": [1, 2, 3, 4],
        # gauss std for filtering spikes beforer loss calculation in ms
        "spike_filter_std": 5,
        # if input_with_state then augment input with state features e.g. water capacity and time else dont
        "input_with_state": 0,
        # if verbose print all messages else no
        "verbose": 1,
        # percentage of hidden neurons(neurons that do not correspond to recordings)
        "hidden_perc": 0.0,
        # if train_per_group average per area and exc/inh population and then calculate loss else no average
        "train_per_group": 1,
        # probability a spike to transmitt to the other areas
        # ratio of exc to inh membrane timeconstant
        "exc_inh_tau_mem_ratio": 2.0,
        # ratio of exc to inh synaptic timeconstant
        "exc_inh_tau_syn_ratio": 1.0,
        # original runs jit_EI_LSNN_areas, simplified runs EI_LSNN_simplified
        "lsnn_version": "original",
        # start and endpoint for reaction_time limits defaults none values
        "reaction_time_limits": None,
        # if varied_delay then varie the delays for inter area connections from n_delay to n_inter_delay with 10 ms intervals
        "varied_delay": 0,
        # number of weight groups
        "rec_groups": 1,
        # if keep_all_input then input goes to all areas else only to first area
        "keep_all_input": 0,
        # seed the training
        "seed": 0,
        # loss over single neuron or first averaging over population
        "loss_neuron_wise": 0,
        # loss over single trial
        "loss_trial_wise": 0,
        "coeff_trial_loss": 1,
        # loss for baseline firing rate
        "loss_firing_rate": 1,
        # simple loss, small experiment
        "simple_loss": 0,
        # change mapping
        "change_mapping": 0,
        # restrict inter inh->exc connections
        "restrict_inter_area_inh": 1,
        # size of filter (to find size in ms you need to multiply * dt)
        "filter_size": 20,
        # propability a neuron to be adaptive
        "prop_adaptive": 0.0,
        # step decay optimizer sheduler, every 500 steps
        "step_decay": 5000,
        # step decay multiplier sheduler
        "step_multiplier": 0.5,
        # spike function type {"bernoulli", "poisson", "deterministic"}
        "spike_function": "bernoulli",
        # data load version, either the spike times version(2) or the raster version (1)
        "load_version": 2,
        # constrain mapping in area and excitatory-inhibitory population
        "constrain_on_area": False,
        "constrain_on_exc": True,
        # if False there is different synaptic transmission delay for intra and inter area synapses
        "no_inter_intra": False,
        # train bias (offset in the v_{rest})
        "train_bias": False,
        # train bias(a multiplicative factor in the membrane noise)
        "train_noise_bias": False,
        # the bias is applied in threshold or the membrane
        "bias_in_mem": True,
        # if true there is a membrane offset per neuron
        "trial_offset": False,
        # number of latent spaces of trial noise
        "latent_space": 2,
        # to train the tau_{adaptation} or not
        "train_adaptation": False,
        "train_delays": False,
        "trial_matching": False,
        # proportion of excitatory neurons
        "p_exc": 0.85,
        # maximum iteration before stopping the early stopping
        "early_stop": 2000,
        "input_timesteps": 1,
        "conductance_based": False,
        "trial_loss_area_specific": True,
        "coeff_trial_fr_loss": 0,
        "geometric_loss": False,
        "resample": 0.02,
        "choose_min_delay": True,
        "new_version": False,
        "motor_areas": [],
        "jaw_neurons": 100,
        "jaw_delay": 40,
        "jaw_min_delay": 12,
        "tau_jaw": 100,
        "mean_fr": 5,
        "jaw_version": 0,
        "trial_types": [0, 1, 2, 3],
        "pca_features": False,
        "n_pca_comp": 5,
        "train_with_jaw": False,
        "temperature": 7.5,
        "latent_new": False,
        "with_behaviour": False,
        "loss_trial_type": False,
        "session_based": False,
        "z_score": True,
        "with_task_splitter": False,
        "jaw_open_loop": False,
        "motor_buffer_size": 40,
        "jaw_tongue": 1,
        "jaw_nonlinear": False,
        "scaling_jaw_in_model": False,
        "balance_trial_types": False,
        "t_trial_gan": True,
        "loss_trial_matched_mle": False,
        "use_logits": False,
        "stim_onsets": [0, 1],
        "p_exc_in": 0.8,
        "v_rest": 0,
        "thr": 0.1,
        "trial_offset_bound": False,
        "v_rest_bound": False,
        "weights_distance_based": False,
        "pop_per_area": 1,
        "trial_mle_with_vae": False,
    }

    default["time"] = time.ctime()
    default["num_areas"] = len(default["areas"])
    default["device"] = torch.device(default["device"])
    default = SimpleNamespace(**default)
    default = scale_function(default)
    return default


# # PseudoData
def config_pseudodata():
    opt = get_default_opt()
    opt.datapath = "./datasets/PseudoData_v17_delay108_onesession"
    opt.load_version = 2
    opt.areas = ["area1", "area2"]
    opt.stim = [4]
    opt.num_areas = len(opt.areas)

    opt.n_units = 500
    opt.n_rnn_in = 2
    opt.start, opt.stop = -0.1, 0.4
    opt.batch_size = 200
    opt.no_inter_intra = True
    opt.train_bias = True
    opt.train_noise_bias = True
    opt.train_adaptation = False
    opt.prop_adaptive = 0.0
    opt.noise_level_list = [0.1 for i in range(len(opt.areas))]
    opt.input_f0 = 5
    # opt.lsnn_version = "srm"
    opt.lsnn_version = "simplified"
    opt.thalamic_delay = 0.004  # * (opt.lsnn_version != "srm")
    opt.tau_list = [10 for i in range(opt.num_areas)]
    opt.exc_inh_tau_mem_ratio = 3.0
    opt.stim_onsets = [0]

    opt.restrict_inter_area_inh = True
    opt.dt = 2
    opt.n_delay = 2
    opt.inter_delay = 4
    opt.rec_groups = 2
    opt.input_filter_size = 1
    opt.spike_filter_std = 12  # miliseconds # int(spike_filter_std / opt.dt)

    opt.trial_offset = False
    opt.latent_space = 5
    opt.latent_new = False
    opt.trial_matching = True
    opt.gan_loss = True
    opt.t_trial_gan = True
    opt.loss_trial_matched_mle = 0
    opt.loss_trial_wise = 1
    opt.loss_neuron_wise = 1
    opt.loss_cross_corr = 0
    opt.coeff_loss = 10
    opt.coeff_firing_rate_distro_reg = 0.0
    opt.coeff_trial_loss = 20
    opt.coeff_cross_corr_loss = 0.0
    opt.early_stop = 8000
    opt.coeff_trial_fr_loss = 0.0
    opt.loss_firing_rate = 0

    opt.w_decay = 0.0
    opt.l1_decay = 0.0

    opt.keep_all_input = 1
    opt.change_mapping = 0
    opt.lr = 0.001
    opt.p_exc = 0.8
    opt.p_exc_in = 1
    opt.input_timesteps = 1
    opt.trial_loss_area_specific = True
    opt.geometric_loss = True
    opt.resample = 2
    opt.new_version = True
    opt.motor_areas = []
    opt.jaw_neurons = 100
    opt.jaw_delay = 40
    opt.tau_jaw = 50
    opt.mean_fr = 5
    opt.jaw_version = 1

    opt.conductance_based = False
    opt.gan_hidden_neurons = 128
    opt.spike_function = "bernoulli"
    opt.trial_types = [0, 1]
    opt.pca_features = False
    opt.n_pca_comp = 5
    opt.with_behaviour = False
    opt.loss_trial_type = False
    opt.with_task_splitter = False
    opt.temperature = 15
    opt.stim_valance *= 2.3
    opt.balance_trial_types = True
    opt.use_logits = False
    opt.v_rest = 0  # -75  #
    opt.thr = 0.1  # -50  #
    opt.trial_offset_bound = False
    opt.v_rest_bound = False
    # opt.weights_distance_based = True
    opt.trial_mle_with_vae = False
    opt.num_areas = len(opt.areas)

    return opt


# # Vahid
def config_vahid():
    opt = get_default_opt()
    opt.datapath = "./datasets/DataFromVahid_expert"
    opt.areas = ["wS1", "wS2", "wM1", "wM2", "ALM", "tjM1"]
    opt.num_areas = len(opt.areas)
    opt.stim = [0, 1]
    opt.stim_valance *= 2.3
    opt.n_units = 1500
    opt.n_rnn_in = 2
    opt.start, opt.stop = -0.2, 1.2
    opt.load_version = 2
    opt.noise_level_list = [0.10 for i in range(len(opt.areas))]
    opt.balance_trial_types = False

    opt.prop_adaptive = 0.0
    opt.input_f0 = 5
    opt.tau_list = [10 for i in range(opt.num_areas)]

    opt.dt = 2
    opt.inter_delay = 4  # miliseconds
    opt.n_delay = 2
    opt.rec_groups = 1
    opt.weights_distance_based = True
    opt.choose_min_delay = True
    opt.input_filter_size = 1

    opt.spike_filter_std = 12  # miliseconds # int(spike_filter_std / opt.dt)
    opt.lsnn_version = "simplified"
    opt.thalamic_delay = 0.004  # * (opt.lsnn_version != "srm")
    opt.early_stop = 8000
    opt.lr = 1e-3
    opt.p_exc = 0.8
    opt.p_exc_in = 1
    opt.batch_size = 150

    opt.loss_neuron_wise = 1
    opt.loss_cross_corr = 0
    opt.loss_trial_wise = 1
    opt.loss_firing_rate = 0
    opt.trial_matching = True
    opt.coeff_loss = 10
    opt.coeff_firing_rate_distro_reg = 0.0
    opt.coeff_trial_fr_loss = 0.0000
    opt.coeff_trial_loss = 100
    opt.coeff_cross_corr_loss = 0.0

    opt.trial_offset = False
    opt.latent_space = 5
    opt.no_inter_intra = True
    opt.train_noise_bias = True
    opt.train_adaptation = False
    opt.train_bias = True
    opt.change_mapping = False
    opt.spike_function = "bernoulli"
    opt.w_decay = 0.0
    opt.l1_decay = 0.0

    opt.exc_inh_tau_mem_ratio = 3.0
    opt.keep_all_input = 1
    opt.conductance_based = False
    opt.trial_loss_area_specific = True
    opt.geometric_loss = True
    opt.resample = 2  # stride of already filtered signal
    opt.new_version = True

    opt.motor_areas = [4, 5]
    opt.jaw_neurons = 100
    opt.jaw_delay = 40
    opt.jaw_min_delay = 12
    opt.tau_jaw = 50
    opt.mean_fr = 5
    opt.jaw_version = 1

    opt.gan_loss = False
    opt.gan_hidden_neurons = 128
    opt.latent_new = True
    opt.with_behaviour = True
    # opt.device = "cpu"
    opt.reaction_time_limits = [-1, 0.2]
    opt.with_task_splitter = True
    opt.z_score = True
    opt.jaw_open_loop = True
    opt.scaling_jaw_in_model = True
    opt.motor_buffer_size = 40
    opt.jaw_tongue = 1
    opt.jaw_nonlinear = False
    opt.temperature = 7.5
    opt.v_rest = 0  # -75  #
    opt.thr = 0.1  # -50  #
    opt.trial_offset_bound = False
    opt.v_rest_bound = True

    opt.num_areas = len(opt.areas)
    return opt


def get_opt(log_path=None):
    default = get_default_opt()
    opt = copy.copy(default)
    if log_path == None:
        return opt
    else:
        try:
            opt = load_training_opt(log_path)
            for key in vars(default).keys():
                if vars(opt).get(key) is None:
                    setattr(opt, key, getattr(default, key))

            return opt
        except:
            opt = load_opt_old(log_path)

            for key in vars(default).keys():
                if vars(opt).get(key) is None:
                    setattr(opt, key, getattr(default, key))

            return opt


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--config", type="string", default="none")
    (pars, _) = parser.parse_args()

    config_path = os.path.join("configs", pars.config)
    if ~os.path.exists(config_path):
        os.mkdir(config_path)

    default = get_default_opt()
    opt = copy.copy(default)
    opt = config_pseudodata(opt)

    save_opt(config_path, opt)
