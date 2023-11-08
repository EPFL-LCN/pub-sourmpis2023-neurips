from infopath.session_stitching import build_network
from models.pop_rsnn import PopRSNN
from models.rsnn import RSNN
from datasets.prepare_input import InputSpikes
import numpy as np
import torch.nn as nn
import torch
from utils.logger import reload_weights
import os
import copy
from geomloss import SamplesLoss
from infopath.loss_splitter import NormalizedMultiTaskSplitter
from infopath.losses import *


class FullModel(nn.Module):
    def __init__(self, opt):
        super(FullModel, self).__init__()
        self.opt = opt
        self.num_areas = opt.num_areas
        self.rsnn = init_rsnn(opt)
        self.input_spikes = InputSpikes(opt, opt.input_seed)
        start, stop = self.opt.start, self.opt.stop
        self.opt.start, self.opt.stop = 0.0, 0.3
        self.input_spikes_pre = InputSpikes(copy.deepcopy(opt), opt.input_seed)
        self.opt.start, self.opt.stop = start, stop
        self.timestep = self.opt.dt * 0.001
        self.trial_onset = -int(self.opt.start / self.timestep)
        # some filter functions
        kernel_size1 = int(self.opt.spike_filter_std / self.opt.dt)
        padding = int((kernel_size1 - 1) / 2)
        self.filter1 = torch.nn.AvgPool1d(
            kernel_size1,
            int(kernel_size1 // 2),
            padding=padding,
            count_include_pad=False,
        )
        stride2 = opt.resample
        padding = int((stride2 * 2 - 1) / 2)
        self.filter2 = torch.nn.AvgPool1d(
            2 * stride2, stride2, padding=padding, count_include_pad=False
        )

        self.T = int((self.opt.stop - self.opt.start) / self.opt.dt * 1000)
        input_dimD = int(np.round(self.T / stride2 / int(kernel_size1 // 2)))
        self.input_dimD = input_dimD
        self.jaw_mean = torch.nn.Parameter(torch.zeros(1))
        self.jaw_std = torch.nn.Parameter(torch.ones(1) * 0.1)

        # for the trial-matching loss function
        if opt.geometric_loss:
            self.trial_loss_fun = SamplesLoss(loss="sinkhorn", p=1, blur=0.01)
        else:
            self.trial_loss_fun = hard_trial_matching_loss
        # setup the loss splitter
        task_splitters = opt.loss_trial_wise
        task_splitters += opt.loss_neuron_wise
        task_splitters += opt.loss_cross_corr
        task_splitters += opt.loss_firing_rate
        task_splitters += opt.loss_trial_matched_mle
        self.multi_task_splitter = NormalizedMultiTaskSplitter(task_splitters)

    def filter_fun1(self, spikes):
        """filter spikes

        Args:
            spikes (torch.tensor): spikes with dimensionality time x trials x neurons
        Return:
            filtered spikes with kernel specified from the self.opt
        """
        if spikes is None:
            return None
        spikes = spikes.permute(2, 1, 0)
        spikes = self.filter1(spikes)
        return spikes.permute(2, 1, 0)

    def filter_fun2(self, x):
        """filter signal

        Args:
            x (torch.tensor): signal with dimensionality time x trials x neurons
        Return:
            filtered signals with kernel specified from the self.opt
        """
        if x is None:
            return None
        x = x.permute(2, 1, 0)
        x = self.filter2(x)
        return x.permute(2, 1, 0)

    @torch.no_grad()
    def steady_state(self, state=None, sample_trial_noise=True):
        """
        set state of network in the steady state (basically run the network for 300ms
        from a zero state) we don't train this part so torch.no_grad
        """
        trials = torch.zeros(self.opt.batch_size).long().to(self.opt.device)
        spike_data = self.input_spikes_pre(trials)
        if state is None:
            state = self.rsnn.zero_state(self.opt.batch_size)
        _, _, _, state = self.rsnn(
            spike_data, state, sample_trial_noise=sample_trial_noise
        )
        return state

    def step(self, input_spikes, state, mem_noise, start=None, stop=None):
        opt = self.opt
        if start is None:
            start = 0
        if stop is None:
            stop = input_spikes.shape[0]
        self.rsnn.mem_noise = mem_noise[start:stop]
        spike_outputs, voltages, model_jaw, state = self.rsnn(
            input_spikes[start:stop].to(opt.device),
            state,
            sample_trial_noise=False,
            sample_mem_noise=False,
        )
        if not opt.scaling_jaw_in_model:
            model_jaw = (model_jaw - self.jaw_mean) / self.jaw_std
            if self.opt.jaw_nonlinear:
                model_jaw = torch.exp(model_jaw) - 1
        return spike_outputs, voltages, model_jaw, state

    def forward(self, stims, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.opt.batch_size = stims.shape[0]
        state = self.steady_state()
        input_spikes = self.input_spikes(stims)
        self.rsnn.sample_mem_noise(self.T, stims.shape[0])
        mem_noise = self.rsnn.mem_noise.clone()
        return self.step(input_spikes, state, mem_noise)

    def step_with_dt(
        self, input_spikes, state, light=None, dt=25, sample_mem_noise=False
    ):  # not really used in the paper, usefull for running forward the network without training in smaller gpus
        if sample_mem_noise:
            self.rsnn.sample_mem_noise(input_spikes.shape[0], input_spikes.shape[1])
        mem_noise = self.rsnn.mem_noise.clone()
        spikes, voltages, jaw, l = [], [], [], None
        for i in range(np.ceil(input_spikes.shape[0] / dt).astype(int)):
            self.rsnn.mem_noise = mem_noise[i * dt : (i + 1) * dt].clone()
            if light is not None:
                l = light[i * dt : (i + 1) * dt]
            sp, v, j, state = self.rsnn(
                input_spikes[i * dt : (i + 1) * dt],
                state,
                l,
                sample_mem_noise=False,
                sample_trial_noise=False,
            )
            spikes.append(sp[0])
            voltages.append(v[0])
            jaw.append(j[0])
        spikes = torch.cat(spikes, dim=0)
        voltages = torch.cat(voltages, dim=0)
        jaw = torch.cat(jaw, dim=0)
        self.rsnn.mem_noise = mem_noise
        if not self.opt.scaling_jaw_in_model:
            jaw = (jaw - self.jaw_mean) / self.jaw_std
            if self.opt.jaw_nonlinear:
                model_jaw = torch.exp(model_jaw) - 1
        return spikes, voltages, jaw, state

    # Where we add all the terms of the loss function
    def generator_loss(
        self,
        model_spikes,
        data_spikes,
        model_jaw,
        data_jaw,
        session_info,
        netD,
    ):
        """Calculates the generator loss (in the no GAN like cases is the only/classic loss), if there are multiple loss
        they are weighted with the loss spliter to ensure that their gradients have comparable scales.

        Args:
            model_spikes (torch.tensor): The spikes from the model $z$
            data_spikes (torch.tensor): The spikes from the recordings $z^{\mathcal{D}}$
            model_jaw (torch.tensor, None): The jaw trace from the model
            data_jaw (torch.tensor, None): The jaw trace from the data
            session_info (list): List with information about the different sessions
            netD (torch.nn.Module, None): _description_

        Returns:
            torch.float: Total loss
        """
        count_tasks = 0  # useful for the the loss_splitter
        if self.opt.with_task_splitter:
            model_spikes_list = self.multi_task_splitter(model_spikes)
        else:
            model_spikes_list = [model_spikes]
        opt = self.opt
        # filter once for the $T_{neuron}$ and twice for the $T_{trial}$
        filt_model = self.filter_fun1(model_spikes)
        filt_data = self.filter_fun1(data_spikes)
        filt_model_jaw = self.filter_fun1(model_jaw)
        if data_jaw is not None:
            filt_data_jaw = self.filter_fun1(data_jaw)
        else:
            filt_data_jaw = None
        # Generator loss
        neur_loss, trial_loss, fr_loss, cross_corr_loss, tm_mle_loss = 0, 0, 0, 0, 0
        if opt.loss_firing_rate:  # not used in the paper
            if self.opt.with_task_splitter:
                fr_loss += firing_rate_loss(model_spikes_list[count_tasks])
                count_tasks += 1
            else:
                fr_loss += firing_rate_loss(model_spikes)
        if opt.loss_cross_corr:  # not used in the paper
            if self.opt.with_task_splitter:
                cross_corr_loss += cross_corr_loss(
                    model_spikes_list[count_tasks], data_spikes, session_info
                )
                count_tasks += 1
            else:
                cross_corr_loss += cross_corr_loss(
                    model_spikes, data_spikes, session_info
                )
        if opt.loss_neuron_wise:  # T_{neuron}
            if self.opt.with_task_splitter:
                filt_model = self.filter_fun1(model_spikes_list[count_tasks])
                count_tasks += 1
            neur_loss += neuron_loss(filt_data, filt_model)

        if opt.loss_trial_wise:  # T_{trial}
            if opt.gan_loss:  #  if True for GAN else trial matching
                if self.opt.with_task_splitter:
                    model_spikes = model_spikes_list[count_tasks]
                    count_tasks += 1
                # for the GAN loss usually is good to deactivate the loss_splitter
                if self.opt.t_trial_gan:
                    model_spikes = self.filter_fun2(self.filter_fun1(model_spikes))
                    data_spikes = self.filter_fun2(filt_data)
                    data_jaw = self.filter_fun2(filt_data_jaw)
                    model_jaw = self.filter_fun2(filt_model_jaw)
                trial_loss, _ = discriminator_loss(
                    netD,
                    model_spikes,
                    data_spikes,
                    data_jaw,
                    model_jaw,
                    session_info,
                    self.rsnn.area_index,
                    discriminator=False,
                    t_trial_gan=self.opt.t_trial_gan,
                    z_score=self.opt.z_score,
                    trial_loss_area_specific=self.opt.trial_loss_area_specific,
                )
            else:
                if self.opt.with_task_splitter:
                    filt_model = self.filter_fun1(model_spikes_list[count_tasks])
                    count_tasks += 1
                trial_loss = trial_matching_loss(
                    self.filter_fun2(filt_data),
                    self.filter_fun2(filt_model),
                    session_info,
                    self.filter_fun2(filt_data_jaw),
                    self.filter_fun2(filt_model_jaw),
                    self.trial_loss_fun,
                    self.rsnn.area_index,
                    z_score=self.opt.z_score,
                    trial_loss_area_specific=self.opt.trial_loss_area_specific,
                )
        if opt.loss_trial_matched_mle:
            if self.opt.with_task_splitter:
                filt_model = self.filter_fun1(model_spikes_list[count_tasks])
                count_tasks += 1
            tm_mle_loss += trial_matched_mle_loss(
                self.filter_fun2(filt_data),
                self.filter_fun2(filt_model),
                session_info,
                data_jaw,
                model_jaw,
            )
        firing_rate_loss = opt.coeff_firing_rate_distro_reg * fr_loss
        trial_loss *= opt.coeff_trial_loss
        neur_loss *= opt.coeff_loss
        cross_corr_loss *= opt.coeff_cross_corr_loss
        tm_mle_loss *= opt.coeff_loss
        return firing_rate_loss, trial_loss, neur_loss, cross_corr_loss, tm_mle_loss

    def mean_activity(self, activity, clusters=None):
        with torch.no_grad():
            device = self.opt.device
            activity = self.filter_fun1(activity.to(device)).cpu()
            if clusters is None:
                clusters = torch.arange(activity.shape[2]) > 0
            step = self.timestep
            activity = activity[..., clusters]
            exc_index = self.rsnn.excitatory_index[clusters]
            area = self.rsnn.area_index[clusters]
            mean_exc, mean_inh = [], []
            for i in range(self.num_areas):
                area_index = area == i
                exc_mask = exc_index & area_index
                exc_mask = exc_mask.cpu()
                simulation_exc = (
                    np.nanmean(activity[..., exc_mask].cpu(), (1, 2)) / step
                )
                mean_exc.append(simulation_exc)
                inh_index = ~exc_index
                inh_mask = inh_index & area_index
                inh_mask = inh_mask.cpu()
                simulation_inh = (
                    np.nanmean(activity[..., inh_mask].cpu(), (1, 2)) / step
                )
                mean_inh.append(simulation_inh)
        return mean_exc, mean_inh


def load_model_and_optimizer(opt, reload=False, last_best="last"):
    model = FullModel(opt)
    if os.path.exists(os.path.join(opt.log_path, "sessions.npy")):
        sessions = np.load(
            os.path.join(opt.log_path, "sessions.npy"), allow_pickle=True
        )
        model.neuron_index = np.load(os.path.join(opt.log_path, "neuron_index.npy"))
        model.firing_rate = np.load(os.path.join(opt.log_path, "firing_rate.npy"))
        model.areas = np.load(
            os.path.join(opt.log_path, "areas.npy"), allow_pickle=True
        )
        model.sessions = sessions
    else:
        neuron_index, firing_rate, session, area = build_network(
            model.rsnn, opt.datapath, opt.areas, opt.with_behaviour, opt.hidden_perc
        )
        model.neuron_index = neuron_index
        model.firing_rate = firing_rate
        model.areas = area
        model.sessions = session

    if opt.gan_loss:
        if opt.t_trial_gan:  # for spikeT-GAN
            netD = DiscriminatorSession(
                model, opt.gan_hidden_neurons, opt.with_behaviour
            )
        else:  # for spike-GAN
            netD = DiscriminatorSessionCNN(
                model, opt.gan_hidden_neurons, opt.with_behaviour
            )
        optimizerD = torch.optim.AdamW(netD.parameters(), lr=opt.lr, weight_decay=1)
        netD.to(opt.device)
    else:
        netD, optimizerD = None, None
    optimizerG = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=opt.w_decay
    )

    if reload:
        reload_weights(opt, model, optimizerG, last_best=last_best)
        if opt.gan_loss:
            optim_path = os.path.join(opt.log_path, "last_optimD.ckpt")
            netD_path = os.path.join(opt.log_path, "last_netD.ckpt")
            optimizerD.load_state_dict(
                torch.load(optim_path, map_location=opt.device.type)
            )
            netD.load_state_dict(torch.load(netD_path, map_location=opt.device.type))

    model.to(opt.device)
    return model, netD, optimizerG, optimizerD


def init_rsnn(opt):
    if opt.lsnn_version != "mean":
        rsnn = RSNN(
            opt.n_rnn_in,
            opt.n_units,
            sigma_mem_noise=opt.noise_level_list,
            num_areas=opt.num_areas,
            tau_adaptation=opt.tau_adaptation,
            tau=opt.tau_list,
            exc_inh_tau_mem_ratio=opt.exc_inh_tau_mem_ratio,
            n_delay=opt.n_delay,
            inter_delay=opt.inter_delay,
            restrict_inter_area_inh=opt.restrict_inter_area_inh,
            prop_adaptive=opt.prop_adaptive,
            dt=opt.dt,
            p_exc=opt.p_exc,
            spike_function=opt.spike_function,
            train_v_rest=opt.train_bias,
            trial_offset=opt.trial_offset,
            rec_groups=opt.rec_groups,
            latent_space=opt.latent_space,
            train_adaptation=opt.train_adaptation,
            train_noise_bias=opt.train_noise_bias,
            conductance_based=opt.conductance_based,
            jaw_delay=opt.jaw_delay,
            jaw_min_delay=opt.jaw_min_delay,
            tau_jaw=opt.tau_jaw,
            motor_areas=opt.motor_areas,
            temperature=opt.temperature,
            latent_new=opt.latent_new,
            jaw_open_loop=opt.jaw_open_loop,
            scaling_jaw_in_model=opt.scaling_jaw_in_model,
            p_exc_in=opt.p_exc_in,
            v_rest=opt.v_rest,
            thr=opt.thr,
            trial_offset_bound=opt.trial_offset_bound,
        )
    else:
        rsnn = PopRSNN(
            opt.n_rnn_in,
            opt.n_units,
            sigma_mem_noise=opt.noise_level_list,
            num_areas=opt.num_areas,
            tau=opt.tau_list,
            n_delay=opt.n_delay,
            inter_delay=opt.inter_delay,
            dt=opt.dt,
            p_exc=opt.p_exc,
            rec_groups=opt.rec_groups,
            latent_space=opt.latent_space,
            train_noise_bias=opt.train_noise_bias,
            jaw_delay=opt.jaw_delay,
            jaw_min_delay=opt.jaw_min_delay,
            tau_jaw=opt.tau_jaw,
            motor_areas=opt.motor_areas,
            latent_new=opt.latent_new,
            jaw_open_loop=opt.jaw_open_loop,
            scaling_jaw_in_model=opt.scaling_jaw_in_model,
            p_exc_in=opt.p_exc_in,
            trial_offset=opt.trial_offset,
            pop_per_area=opt.pop_per_area,
        )
    return rsnn


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_neurons) -> None:
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, int(hidden_neurons // 2))
        self.fc3 = nn.Linear(int(hidden_neurons // 2), 1)

    def forward(self, x):
        x = torch.nn.LeakyReLU(0.2)(self.fc1(x))
        x = torch.nn.LeakyReLU(0.2)(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class DiscriminatorSession(nn.Module):
    def __init__(self, model, hidden_neurons, with_behaviour=True):
        super(DiscriminatorSession, self).__init__()
        self.discriminators = torch.nn.ModuleList()
        sessions = np.unique(model.sessions)
        input_dim = model.input_dimD - 1
        for session in sessions:
            n_areas = num_areas(model, session) + with_behaviour * 1
            self.discriminators.append(
                Discriminator(input_dim * n_areas, hidden_neurons)
            )


class DiscriminatorCNN(nn.Module):
    def __init__(self, input_dim, hidden_neurons):
        super(DiscriminatorCNN, self).__init__()
        kernel_size_1, dilation1, stride1, padding1 = 6, 1, 3, 0
        kernel_size_2, dilation2, stride2, padding2 = 8, 1, 4, 0
        # kernel_size_1, dilation1, stride1, padding1 = 4, 1, 2, 0
        # kernel_size_2, dilation2, stride2, padding2 = 4, 1, 2, 0
        self.num_neurons = input_dim[1]
        time_points = input_dim[0]
        filters1, filters2 = 32, 32
        self.conv1 = nn.Conv1d(
            self.num_neurons,
            filters1,
            kernel_size=kernel_size_1,
            dilation=dilation1,
            padding=padding1,
            stride=stride1,
        )
        nn.init.xavier_uniform_(self.conv1.weight)
        lout_conv1 = int(
            (time_points + 2 * padding1 - dilation1 * (kernel_size_1 - 1) - 1) / stride1
            + 1
        )

        self.batchnorm1 = nn.BatchNorm1d(filters1)
        self.conv2 = nn.Conv1d(
            filters1,
            filters2,
            kernel_size=kernel_size_2,
            dilation=dilation2,
            padding=padding2,
            stride=stride2,
        )
        nn.init.xavier_uniform_(self.conv2.weight)
        lout_conv2 = int(
            (lout_conv1 + 2 * padding2 - dilation2 * (kernel_size_2 - 1) - 1) / stride2
            + 1
        )

        self.batchnorm2 = nn.BatchNorm1d(filters2)
        self.layernorm1 = nn.LayerNorm(filters1)
        self.layernorm2 = nn.LayerNorm(filters2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear((lout_conv2) * filters2, hidden_neurons)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(hidden_neurons, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.dropout = nn.Dropout()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class DiscriminatorSessionCNN(nn.Module):
    def __init__(self, model, hidden_neurons, with_behaviour=True):
        super(DiscriminatorSessionCNN, self).__init__()
        self.discriminators = torch.nn.ModuleList()
        sessions = np.unique(model.sessions)
        for session in sessions:
            neurons = (model.sessions == session).sum() + with_behaviour
            self.discriminators.append(
                DiscriminatorCNN([model.T, neurons], hidden_neurons)
            )


def num_areas(model, session):
    area_index = model.rsnn.area_index
    areas = area_index[session == model.sessions].unique()
    num_areas = len(areas)
    for area in areas:
        if (area_index[session == model.sessions] == area).sum() < 10:
            num_areas -= 1
    return num_areas
