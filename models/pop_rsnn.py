import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from models.rec_weight_matrix import *


class PopRSNN(nn.Module):
    def __init__(
        self,
        input_size,
        n_units,
        pop_per_area=2,
        dt=1.0,
        tau=[10.0],
        n_delay=5,
        inter_delay=20,
        num_areas=3,
        p_exc=0.85,
        varied_delay=False,
        choose_min_delay=True,
        trial_offset=False,
        rec_groups=1,
        latent_space=2,
        train_noise_bias=False,
        train_taumem=False,
        motor_areas=[],
        jaw_delay=40,
        jaw_min_delay=0,
        tau_jaw=100,
        latent_new=False,
        jaw_open_loop=False,
        scaling_jaw_in_model=False,
        p_exc_in=None,
        sigma_mem_noise=[0.1],
    ):
        """Run conductanced-based LIF neurons with synaptic delays

        Args:
            input_size (int): number of input neurons
            n_units (int): number of rec neurons
            tau_adaptation (float, optional): timeconstant of adaptive threshold. Defaults to 144.0.
            beta (float, optional): increase of threshold after spike. Defaults to 1.6.
            v_rest (float, optional): resting potential. Defaults to 0.
            thr (float, optional): resting threshold. Defaults to 0.1.
            prop_adaptive (float, optional): propability of neuron being adaptive. Defaults to 0.2.
            dt (float, optional): timestep in ms. Defaults to 1.0.
            tau (list, optional): membrane time constant per area. Defaults to [10.0, 10, 10].
            exc_inh_tau_mem_ratio (int, optional): how bigger is the excitatory timeconstant to comparison with the inh. Defaults to 2.
            n_refractory (int, optional): number of ms that the neurons are refractory. Defaults to 5.
            n_delay (int, optional): number of ms for the shorter synaptic delay, this number is important also for the implementation of the forward path. Defaults to 5.
            inter_delay (int, optional): number of ms of the longest delay. Defaults to 20.
            dampening_factor (float, optional): scaling of the surrogate derivative. Defaults to 0.3.
            num_areas (int, optional): number of areas. Defaults to 3.
            p_exc (float, optional): proportion of excitatory to inhibitory neurons. Defaults to 0.85.
            p_inpe (int, optional): proportion of possible active input synapses from thalamus to exc. Defaults to 1.
            p_inpi (int, optional): proportion of possible active input synapses from thalamus to exc. Defaults to 1.
            p_ee (float, optional): proportion of possible active synapses from exc to exc. Defaults to 0.2.
            p_ei (float, optional): proportion of possible active synapses from exc to inh. Defaults to 0.6.
            p_ie (float, optional): proportion of possible active synapses from inh to exc. Defaults to 0.6.
            p_ii (float, optional): proportion of possible active synapses from inh to inh. Defaults to 0.6.
            prop_light (float) : probability of one inhibitory neuron has light-activated channels, frac{\DeltaF}{F} > 0.5, actual value from ws1 data 48/105
            sigma_mem_noise (list, optional): noise level per area. Defaults to [0.16, 0.12, 0.12].
            temperature (float, optional): parameter for no deterministic spike_function. Defaults to 7.5.
            varied_delay (bool, optional): if the delays are varied. Defaults to False.
            keep_all_input (bool, optional): input projects to all areas (True) input projects only to first (False). Defaults to False.
            restrict_inter_area_inh (bool, optional): inhibitory neurons project to other areas (False). Defaults to False.
            spike_function (str, optional): spike_function. Defaults to "bernoulli".
            train_v_rest (bool, optional): scaling of the noise level with a trained parameter. Defaults to False.
            trial_offset (bool, optional): constant per trial noise, if True the an array of (latent space)x(num neurons) dim is trained. Defaults to False.
            rec_groups (int, optional): number of recurrent weight matrices (each matrix have different delays). Defaults to 1.
            latent_space (int, optional): Look trial_offset. Defaults to 2.
            train_adaptation (bool, optional): if the adaptation is trained or not. Defaults to False.
        """
        super(PopRSNN, self).__init__()
        self.input_size = input_size
        self.n_units = n_units
        self.scaling_jaw_in_model = scaling_jaw_in_model
        # the convention is that we always start with excitatory
        self.excitatory = int(p_exc * n_units)
        self.inhibitory = n_units - self.excitatory
        self.num_areas = num_areas
        self.latent_size = num_areas * pop_per_area
        self.ppa = pop_per_area
        self.p_exc = p_exc
        self.p_exc_in = p_exc if p_exc_in is None else p_exc_in
        self.n_delay = int(n_delay // dt)
        self.motor_areas = torch.tensor(motor_areas)
        self.init_population_indices()
        self.v_rest = Parameter(torch.zeros(self.latent_size))
        self.tau_jaw = tau_jaw
        self.decay_jaw = torch.exp(torch.tensor(-dt / self.tau_jaw))

        self.tau = torch.ones(self.latent_size) * tau[0]
        if train_taumem:
            self.decay_v = Parameter(torch.exp(-dt / self.tau.clone().detach()))
        else:
            self.register_buffer("decay_v", torch.exp(-dt / self.tau.clone().detach()))

        self.jaw_delay = int(jaw_delay // dt)
        self.jaw_min_delay = int(jaw_min_delay // dt)
        self.jaw_open_loop = jaw_open_loop
        if self.motor_areas.shape[0] > 0:
            self.jaw_kernel = self.jaw_delay - self.jaw_min_delay + 1 - self.n_delay
            self._w_jaw_pre = Parameter(torch.zeros(self.motor_area_index.sum(), 1))
            self._w_jaw_post = Parameter(torch.zeros(self.jaw_kernel, self.n_units))
            torch.nn.init.xavier_normal_(self._w_jaw_pre)
            torch.nn.init.xavier_normal_(self._w_jaw_post)
            self.conv = torch.nn.Conv1d(
                1, self.n_units, kernel_size=self.jaw_kernel, bias=False
            )
            self._w_jaw_pre.data *= 10
            self.conv.weight.data = self._w_jaw_post[None].permute(2, 0, 1)
            self.jaw_bias = Parameter(torch.ones(1))

        weights_in = torch.randn(self.latent_size, input_size).detach()
        weights_in /= self.input_size**0.5
        self._w_in = Parameter(weights_in)

        self.rec_groups = rec_groups
        weights = []
        for i in range(self.rec_groups):
            weights_rec = torch.randn(self.latent_size, self.latent_size).detach()
            weights.append(weights_rec / self.latent_size**0.5)
        weights_rec = torch.stack(weights) / self.rec_groups

        self._w_rec = Parameter(weights_rec)
        self.dt = dt
        self.projections = nn.ModuleList(
            [
                nn.Linear(pop_per_area, (self.area_index == 0).sum())
                for i in range(num_areas)
            ]
        )
        self.sigma_mem_noise = sigma_mem_noise[0]

        self.inter_delay = int(inter_delay // dt)
        self.varied_delay = varied_delay

        delays = [np.ones(self.latent_size) * (self.inter_delay - self.n_delay)]
        if (rec_groups == 1) and (not choose_min_delay):
            delays = [np.zeros(self.latent_size)]
        if rec_groups > 1:
            num_delay = (self.inter_delay - self.n_delay) / (rec_groups - 1)
            delays += [
                np.ones(self.latent_size) * int(num_delay * i)
                for i in range(rec_groups - 1)
            ]
        delays = np.array(delays).astype(int).T
        self.delays = torch.tensor(delays)
        self.mask = self.mask_delay()

        if train_noise_bias:
            self.bias = Parameter(torch.ones(self.latent_size))
        else:
            self.register_buffer(
                "bias", torch.ones(self.latent_size, requires_grad=False)
            )
        self.latent_space = latent_space
        self.latent_new = latent_new
        if latent_new:
            self.lin_tr_offset = torch.nn.Linear(latent_space, latent_space)
            self.lin1_tr_offset = torch.nn.Linear(
                latent_space, self.latent_size, bias=False
            )
        if trial_offset:
            trial_offset_init = (
                torch.rand(latent_space, self.latent_size) - 0.5
            ) / latent_space**0.5
            self.trial_offset = Parameter(trial_offset_init)
        else:
            trial_offset_init = torch.zeros(latent_space, self.latent_size)
            self.register_buffer("trial_offset", trial_offset_init)

    def init_population_indices(self):
        num_areas = self.num_areas
        exc = self.excitatory = int(self.p_exc * self.n_units)
        inh = self.inhibitory = self.n_units - self.excitatory
        pop = torch.zeros(self.n_units, num_areas * 2)
        self.register_buffer("population", pop)
        for i in range(num_areas):
            start_exc = i * exc // num_areas
            stop_exc = (i + 1) * exc // num_areas if i != num_areas else exc
            self.population[start_exc:stop_exc, i] = 1
            start_inh = exc + i * inh // num_areas
            stop_inh = exc + (i + 1) * inh // num_areas if i != num_areas else exc + inh
            self.population[start_inh:stop_inh, i + num_areas] = 1
        self.population = self.population > 0
        area_index = torch.argmax(self.population * 1.0, dim=1) % num_areas
        self.register_buffer("area_index", area_index)
        exc_index = torch.argmax(self.population * 1.0, dim=1) < num_areas
        self.register_buffer("excitatory_index", exc_index)
        self.motor_area_index = torch.isin(area_index, self.motor_areas)
        self.motor_area_index *= self.excitatory_index

    def mask_delay(self):
        """It is combined with self.delays to make the buffer slicing"""
        delays = self.delays
        mask = torch.zeros(self.inter_delay, self.latent_size, delays.shape[1])
        mask = mask.type(torch.bool).to(self._w_rec.device)
        for i, delay in enumerate(delays):
            for j, d in enumerate(delay):
                mask[d : d + self.n_delay, i, j] = 1
        return mask

    def sample_trial_noise(self, batch_size):
        device = self._w_in.device
        self.trial_noise = torch.rand(batch_size, self.latent_space, device=device)
        self.trial_noise -= 0.5

    def sample_mem_noise(self, input_time, batch_size):
        device = self._w_in.device
        self.mem_noise = torch.randn(
            input_time, batch_size, self.latent_size, device=device
        )

    def prepare_currents(self, spike_buffer):
        buffer = []
        for g in range(self.rec_groups):
            i, k = torch.where(self.mask[..., g].T)
            buf = spike_buffer[k, :, i].reshape(self.latent_size, self.n_delay, -1)
            buf = buf.permute(1, 2, 0)
            buffer.append(buf)
        buffer = torch.stack(buffer)
        rec_cur = torch.einsum("gtbi,gji->tbj", buffer, self._w_rec)
        return rec_cur

    def forward(
        self,
        input,
        state,
        light=None,
        sample_mem_noise=True,
        sample_trial_noise=True,
        seed=None,
    ):
        spike_buffer, v, jaw_buffer = state[0]

        if sample_mem_noise:
            self.sample_mem_noise(input.shape[0], input.shape[1])
        if sample_trial_noise:
            self.sample_trial_noise(input.shape[1])
        if self.latent_new:
            self.trial_noise_ready = self.lin1_tr_offset(
                torch.relu(self.lin_tr_offset(self.trial_noise))
            )
        else:
            self.trial_noise_ready = self.trial_noise @ self.trial_offset

        inp_cur_total = torch.einsum("tbi,ji->tbj", input, self._w_in)

        mem_noise = self.mem_noise.clone() * self.bias
        if seed is not None:
            torch.manual_seed(seed)
        spike_list = []
        voltage_list = []
        jaw_list = []
        for i in range(input.shape[0] // self.n_delay):
            rec_cur_dt = self.prepare_currents(spike_buffer)
            if self.motor_areas.shape[0] > 0:
                active_buffer = jaw_buffer[
                    : self.jaw_kernel + self.n_delay - 1
                ].permute(1, 2, 0)
                if not self.jaw_open_loop:
                    rec_cur_jaw = self.conv(active_buffer).permute(2, 0, 1)
            for t_idx in range(self.n_delay):
                abs_t_idx = i * self.n_delay + t_idx
                rec_cur = [rec_cur_dt[t_idx]]
                if self.motor_areas.shape[0] > 0 and not self.jaw_open_loop:
                    rec_cur.append(rec_cur_jaw[t_idx])
                inp_cur = [inp_cur_total[abs_t_idx]]
                z, v = self.step(rec_cur, inp_cur, v, mem_noise[abs_t_idx])
                jaw_buffer = self.step_jaw(z, jaw_buffer)
                jaw_list.append(jaw_buffer[-1])
                voltage_list.append(v[0])
                spike_list.append(z[0])
                spike_buffer = torch.cat([spike_buffer[1:], v])
        state = spike_buffer, v, jaw_buffer
        jaw = torch.stack(jaw_list)
        voltages = torch.stack(voltage_list)
        spikes = torch.stack(spike_list)

        return [spikes], [voltages], [jaw], [state]

    def step(self, rec_cur, inp_cur, v, mem_noise):
        cur = inp_cur[0] + rec_cur[0]
        if torch.isnan(v).sum() > 0:
            print("whh")
        if self.motor_areas.shape[0] > 0 and not self.jaw_open_loop:
            cur += rec_cur[1]
        cur = torch.tanh(cur)
        mem_noise *= self.sigma_mem_noise * self.dt**0.5

        v = (
            self.decay_v * v
            + (1 - self.decay_v) * (cur + self.trial_noise_ready + self.v_rest)
            + mem_noise
        )
        z = torch.zeros(1, v.shape[1], self.n_units, device=v.device)
        for i in range(self.num_areas):
            z[0, :, self.area_index == i] = self.projections[i](
                v[0, :, i * self.ppa : (i + 1) * self.ppa]
            )
        z = self.spike_function(z - 5)
        return z, v

    def spike_function(self, x, dampening_factor=1):
        z_backward = torch.sigmoid(x) * self.dt / 2
        z_forward = (torch.rand_like(x) < z_backward).float()
        z_backward = z_backward * dampening_factor
        z = (z_forward - z_backward).detach() + z_backward
        return z

    def step_jaw(self, z, jaw_buffer):
        if self.motor_areas.shape[0] > 0:
            inp = z[:, :, self.motor_area_index] / 2
            inp = inp @ self._w_jaw_pre
            jpre = jaw_buffer[-1]
            if self.scaling_jaw_in_model:
                jpre = torch.log(jpre + self.jaw_bias)
            j = self.decay_jaw * jpre + (1 - self.decay_jaw) * inp
            if self.scaling_jaw_in_model:
                j = torch.exp(j) - self.jaw_bias
            jaw_buffer = torch.cat([jaw_buffer[1:], j])
        return jaw_buffer

    def zero_state(self, batch_size):
        device = self._w_rec.device
        spikes0 = torch.zeros(
            size=[max(self.inter_delay, self.n_delay), batch_size, self.latent_size],
            device=device,
        )
        voltage0 = torch.zeros(size=[1, batch_size, self.latent_size], device=device)
        jaw_buffer0 = torch.zeros(
            size=[self.jaw_delay + self.n_delay - 1, batch_size, 1], device=device
        )
        state0 = (spikes0, voltage0, jaw_buffer0)
        return state0

    def reform_recurent(self, lr, l1_decay=0.01):
        pass

    def reform_v_rest(self):
        pass
