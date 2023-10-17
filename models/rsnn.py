import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from models.rec_weight_matrix import *


class RSNN(nn.Module):
    def __init__(
        self,
        input_size,
        n_units,
        tau_adaptation=144.0,
        beta=1.6,
        v_rest=0,
        thr=0.1,
        prop_adaptive=0.2,
        dt=1.0,
        tau=[10.0, 10, 10],
        exc_inh_tau_mem_ratio=2,
        n_refractory=4,
        n_delay=5,
        inter_delay=20,
        dampening_factor=0.3,
        num_areas=3,
        p_exc=0.85,
        p_inpe=1,
        p_inpi=1,
        p_ee=1,
        p_ei=1,
        p_ie=1,
        p_ii=1,
        prop_light=0.51,
        sigma_mem_noise=[0.1, 0.1, 0.1],
        temperature=15,
        restrict_inter_area_inh=True,
        spike_function="bernoulli",
        train_v_rest=False,
        trial_offset=False,
        rec_groups=1,
        latent_space=2,
        train_adaptation=False,
        train_noise_bias=False,
        conductance_based=False,
        train_taumem=False,
        motor_areas=[],
        jaw_delay=40,
        jaw_min_delay=0,
        tau_jaw=100,
        latent_new=False,
        jaw_open_loop=False,
        scaling_jaw_in_model=False,
        p_exc_in=None,
        trial_offset_bound=False,
    ):
        """The module for the RSNN. In this function the 2 most complicated parts are: 1. the initialization and the reforming
        of the weight matrices, in order to keep the excitatory and inhibitory neuron types, and, 2. the time simulation in
        particular how we implement the synaptic delays efficiently. This leads to the spike buffers in the state.

        Args:
            input_size (int): number of input neurons/nodes
            n_units (int): number of recurrent units
            tau_adaptation (float, optional): Timeconstant of adaptation in ms. Defaults to 144.0.
            beta (float, optional): Strength of adaptation. Defaults to 1.6.
            v_rest (float, optional): value of membrane resting potential. Defaults to 0.
            thr (float, optional): value of threshold membrane potential. Defaults to 0.1.
            prop_adaptive (float, optional): Probability of a neuron being adaptive. Defaults to 0.2.
            dt (float, optional): timestep of simulatioun. Defaults to 1.0.
            tau (list, optional): list of the membrane timeconstants per area for inhibitory neurons. Defaults to [10.0, 10, 10].
            exc_inh_tau_mem_ratio (int, optional): ratio of excitatory to inhibitory membrane timeconstant. Defaults to 2.
            n_refractory (int, optional): Refractory time period in ms. Defaults to 4.
            n_delay (int, optional): Minimum synaptic delay in ms. Defaults to 5.
            inter_delay (int, optional): maximum synaptic delay in ms. Defaults to 20.
            dampening_factor (float, optional): Rescaling factor for the gradient of the spiking function. Defaults to 0.3.
            num_areas (int, optional): Number of brain areas. Defaults to 3.
            p_exc (float, optional): Probability of a neuron being excitatory. Defaults to 0.85.
            p_inpe (int, optional): probability of . Defaults to 1.
            p_inpi (int, optional): _description_. Defaults to 1.
            p_ee (int, optional): _description_. Defaults to 1.
            p_ei (int, optional): _description_. Defaults to 1.
            p_ie (int, optional): _description_. Defaults to 1.
            p_ii (int, optional): _description_. Defaults to 1.
            prop_light (float, optional): Probability of a neuron being opto sensitive. Defaults to 0.51.
            sigma_mem_noise (list, optional): Strength of gaussian membrane noise as a ratio of (v_thr-v_rest) distance. Defaults to [0.1, 0.1, 0.1].
            temperature (int, optional): Temperature of the sigmoid, when the spike function is "bernoulli". Defaults to 15.
            restrict_inter_area_inh (bool, optional): _description_. Defaults to True.
            spike_function (str, optional): String of the possible spike_functions {"bernoulli", "deterministic"}. Defaults to "bernoulli".
            train_v_rest (bool, optional): Whether to train the v_rest. Defaults to False.
            trial_offset (bool, optional): Wheter to allow for a trial specific bias. Defaults to False.
            rec_groups (int, optional): Number of recurrent weight matrices. Defaults to 1.
            latent_space (int, optional): Number of latent variables for the trial_offset. Defaults to 2.
            train_adaptation (bool, optional): Whether to train the adaptation parameter. Defaults to False.
            train_noise_bias (bool, optional): Wheter to train the strength of the noise. Defaults to False.
            conductance_based (bool, optional): Whether the model to be conductance based. Defaults to False.
            train_taumem (bool, optional): Wheter to train the membrane timeconstant. Defaults to False.
            motor_areas (list, optional): Which areas to use for the jaw linear readout. Defaults to [].
            jaw_delay (int, optional): How long in the past spikes can affect the jaw trace, in ms. Defaults to 40.
            jaw_min_delay (int, optional): Kind of Synaptic delay to motor neurons. Defaults to 0.
            tau_jaw (int, optional): Timeconstant of the jaw trace. Defaults to 100.
            latent_new (bool, optional): Which type of trial_offset generation to use. Defaults to False.
            jaw_open_loop (bool, optional): Whether the jaw to project back to the RSNN. Defaults to False.
            scaling_jaw_in_model (bool, optional): Whether to scale the jaw in the model. Defaults to False.
            p_exc_in (float, optional): Probability of input neuron being excitatory. Defaults to None.
            trial_offset_bound (bool, optional): Whether to allow the trial_offset to be higher than the v_thr. Defaults to False.
        """
        super(RSNN, self).__init__()
        self.input_size = input_size
        self.n_units = n_units
        self.prop_light = prop_light
        self.thr0 = thr
        self.scaling_jaw_in_model = scaling_jaw_in_model
        self.trial_offset_bound = trial_offset_bound
        # the convention is that we always start with excitatory
        self.excitatory = int(p_exc * n_units)
        self.inhibitory = n_units - self.excitatory
        self.num_areas = num_areas
        self.restrict_inter_area_inh = restrict_inter_area_inh
        self.p_exc = p_exc
        self.p_exc_in = p_exc if p_exc_in is None else p_exc_in
        self.motor_areas = torch.tensor(motor_areas)
        self.init_population_indices()
        self.sigma_mem_noise = self.list_to_population(sigma_mem_noise, num_areas)
        self.conductance_based = conductance_based
        self.v_rest0 = v_rest
        if train_v_rest:
            v1 = (torch.rand(n_units) - 0.5) / n_units**0.5 + v_rest
            self.v_rest = Parameter(v1)
        else:
            v1 = torch.zeros(n_units) + v_rest
            self.register_buffer("v_rest", v1)

        self.tau_jaw = tau_jaw
        self.decay_jaw = torch.exp(torch.tensor(-dt / self.tau_jaw))
        tau = self.list_to_population(tau, num_areas)
        tau[: self.excitatory] *= exc_inh_tau_mem_ratio
        self.register_buffer("tau", tau)
        if train_taumem:
            self.decay_v = Parameter(torch.exp(-dt / self.tau.clone().detach()))
        else:
            self.register_buffer("decay_v", torch.exp(-dt / self.tau.clone().detach()))
        self.default_thr = thr
        self.register_buffer("base_thr", thr * torch.ones(n_units))

        # define the synaptic delay of the weight matrices
        # in the version that exists in the paper, the things are simple since we have two weight
        #   matrices one after the other, with single timestep difference
        # Be aware that in table self.delays 0 means that the spike that affects timestep t0 is in the timestep t0 - inter_delay
        self.n_refractory = int(n_refractory // dt)
        self.inter_delay = int(inter_delay // dt)
        self.dt = dt
        self.n_delay = int(n_delay // dt)
        delays = [np.ones(n_units) * (self.inter_delay - self.n_delay)]
        assert not (
            (rec_groups == 1) and (self.inter_delay != self.n_delay)
        ), "wrong value for the inter_delay"
        delays = [np.ones(n_units) * (self.inter_delay - self.n_delay)]
        if rec_groups > 1:
            # We split the time in rec_groups
            num_delay = (self.inter_delay - self.n_delay) / (rec_groups - 1)
            delays += [
                np.ones(n_units) * int(num_delay * i) for i in range(rec_groups - 1)
            ]
        delays = np.array(delays).astype(int).T
        self.register_buffer("delays", torch.tensor(delays))
        mask = self.mask_delay()
        self.mask = mask

        # Build jaw related weight matrices
        self.jaw_delay = int(jaw_delay // dt)
        self.jaw_min_delay = int(jaw_min_delay // dt)
        self.jaw_open_loop = jaw_open_loop
        if self.motor_areas.shape[0] > 0:
            self._w_jaw_pre = Parameter(torch.zeros(self.motor_area_index.sum(), 1))
            self.jaw_bias = Parameter(torch.ones(1))

            torch.nn.init.xavier_normal_(self._w_jaw_pre)
            # These are needed if the jaw is fed back to the RSNN as input current
            self.jaw_kernel = self.jaw_delay - self.jaw_min_delay + 1 - self.n_delay
            self._w_jaw_post = Parameter(torch.zeros(self.jaw_kernel, self.n_units))
            torch.nn.init.xavier_normal_(self._w_jaw_post)
            self._w_jaw_post.data *= (thr - v_rest) / 0.1
            self.conv = torch.nn.Conv1d(
                1, self.n_units, kernel_size=self.jaw_kernel, bias=False
            )
            self._w_jaw_pre.data *= 10
            self.conv.weight.data = self._w_jaw_post[None].permute(2, 0, 1)

        # make input weight matrix
        weights_in = self.make_input_weight_matrix(p_inpe, p_inpi, self.p_exc_in)
        self.n_exc_inp = int(p_exc_in * self.input_size)
        self._w_in = Parameter(weights_in * (thr - v_rest) / 0.1)

        # make reccurent weight matrix
        self.rec_groups = rec_groups
        weights, mask = [], []
        for i in range(self.rec_groups):
            (
                weights_rec,
                mask_rec,
                mask_inter_inh_exc,
                bounds_rec,
            ) = self.make_rec_weight_matrix(p_ee, p_ei, p_ie, p_ii)
            weights.append(weights_rec)
            mask.append(mask_rec)
        mask_rec = torch.stack(mask)
        weights_rec = torch.stack(weights) / self.rec_groups
        # don't allow self excitation
        self.register_buffer("mask_self_exc", torch.eye(mask_rec.shape[1]))
        # This helps for the deep-rewiring
        self.register_buffer("mask_rec", 1.0 * mask_rec)
        self.register_buffer("mask_inter_inh_exc", mask_inter_inh_exc)
        self.bounds_rec = bounds_rec
        self._w_rec = Parameter(weights_rec * (thr - v_rest) / 0.1)

        # adaptive neurons
        tau_adapt = tau_adaptation * torch.ones(self.n_units)
        if train_adaptation:
            betalist = beta * torch.zeros(self.n_units)
            self.beta = Parameter(betalist)
        else:
            betalist = beta * (prop_adaptive > torch.rand(self.n_units))
            self.register_buffer("beta", betalist)
        self.register_buffer("decay_b", torch.exp(-dt / tau_adapt.clone().detach()))
        self.register_buffer("tau_adaptation", tau_adapt / dt)

        # spike function
        self.dampening_factor = dampening_factor
        temp = torch.tensor(temperature, requires_grad=False)
        self.register_buffer("temperature", temp)
        self.spike_fun_type = spike_function
        if spike_function == "bernoulli":
            self.spike_fun = self.spike_function_bernoulli
        elif spike_function == "deterministic":
            self.spike_fun = self.spike_function
        else:
            assert False, "wrong spike function"

        # strength of gaussian membrane noise
        # the self.bias name is misleading, we kept it for legacy reasons
        self.register_buffer("_sigma_mem_noise", self.sigma_mem_noise.detach().clone())
        self.train_noise_bias = train_noise_bias
        if train_noise_bias:
            self.bias = Parameter(torch.ones(n_units))
        else:
            self.register_buffer("bias", torch.ones(n_units, requires_grad=False))

        # Define if we use a trial-specific bias
        # in order not to have any you need both latent_new==trial_offset==False
        self.latent_space = latent_space
        self.latent_new = latent_new
        if latent_new:
            self.lin_tr_offset = torch.nn.Linear(latent_space, latent_space)
            self.lin1_tr_offset = torch.nn.Linear(latent_space, n_units, bias=False)
            self.lin1_tr_offset.weight.data *= (thr - v_rest) / 0.1
        if trial_offset:
            trial_offset_init = (
                (torch.rand(latent_space, n_units) - 0.5)
                / latent_space**0.5
                * (thr - v_rest)
                / 0.1
            )
            self.trial_offset = Parameter(trial_offset_init)
        else:
            trial_offset_init = torch.zeros(latent_space, n_units)
            self.register_buffer("trial_offset", trial_offset_init)

    def init_population_indices(self):
        """Here we create the arrays {area_index, exc_index, light_neuron, motor_area_index}
        which characterize the type of the neurons
        """
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
        light_neuron = ~self.excitatory_index * self.prop_light
        light_neuron = torch.rand_like(light_neuron) < light_neuron
        self.register_buffer("light_neuron", light_neuron)
        self.motor_area_index = torch.isin(area_index, self.motor_areas)
        self.motor_area_index *= self.excitatory_index

    def list_to_population(self, value, num_areas):
        """From list that refers per area, generate an array that refers per neuron"""
        valuenew = torch.zeros(num_areas * 2)
        for i in range(num_areas):
            valuenew[i] = value[i]
            valuenew[i + num_areas] = value[i]
        valuenew = self.population * 1.0 @ valuenew
        return valuenew

    def mask_delay(self):
        """It is combined with self.delays to make the buffer slicing"""
        n_units = self.n_units
        delays = self.delays
        mask = torch.zeros(self.inter_delay, n_units, delays.shape[1])
        mask = mask.type(torch.bool)
        for i, delay in enumerate(delays):
            for j, d in enumerate(delay):
                mask[d : d + self.n_delay, i, j] = 1
        return mask

    def make_input_weight_matrix(self, p_inpe, p_inpi, p_exc):
        """create the input weight matrix"""
        n_inp = self.input_size
        n_e = self.excitatory
        n_i = self.n_units - n_e
        n_all = self.n_units
        mask_inpe = torch.tensor(
            np.random.choice([0, 1], [n_e, n_inp], p=[1 - p_inpe, p_inpe]),
        )
        mask_inpi = torch.tensor(
            np.random.choice([0, 1], [n_i, n_inp], p=[1 - p_inpi, p_inpi]),
        )
        total_mask = torch.cat([mask_inpe, mask_inpi])
        weights = torch.rand(n_all, n_inp).abs() / (n_inp**0.5)
        weights = weights * total_mask
        if p_exc != 1:
            weights = post_synaptic_weight(weights, p_exc)

        return weights

    def make_mask_with_pei(self, p_ee, p_ei, p_ie, p_ii):
        """Generate mask for allowing specific probabilities for exc-inh connections"""
        n_e = self.excitatory
        n_i = self.inhibitory
        mask_ee = torch.rand(n_e, n_e) < p_ee
        mask_ei = torch.rand(n_i, n_e) < p_ei
        mask_ie = torch.rand(n_e, n_i) < p_ie
        mask_ii = torch.rand(n_i, n_i) < p_ii
        mask = (
            torch.cat(
                [torch.cat([mask_ee, mask_ei], 0), torch.cat([mask_ie, mask_ii], 0)], 1
            )
            * 1.0
        )
        mask -= torch.diag(torch.diag(mask))
        conn_per_bound = [mask_ee.sum(), mask_ei.sum(), mask_ie.sum(), mask_ii.sum()]
        bound_coord = [
            [(0, 0), (n_e, n_e)],  # mask ee
            [(n_e, 0), (n_e + n_i, n_e)],  # mask ei
            [(0, n_e), (n_e, n_e + n_i)],  # mask ie
            [(n_e, n_e), (n_e + n_i, n_e + n_i)],  # mask ii
        ]
        return mask, conn_per_bound, bound_coord

    def make_mask_inter_inh_exc(self):
        """Make mask that if self.restric_inter_area_inh blocks inh connections to other areas"""
        n_e = self.excitatory
        if self.restrict_inter_area_inh:
            mask = torch.zeros(self.n_units, self.n_units)
            # all exc - exc are available
            mask[:, :n_e] = 1
            # allow only intra area inh - exc/inh
            for i in range(self.num_areas):
                area_vector = (self.area_index == i) * 1.0
                mask_area = area_vector[:, None] @ area_vector[None]
                mask[mask_area > 0] = 1
        else:
            mask = torch.ones(self.n_units, self.n_units)
        return mask

    def make_rec_weight_matrix(self, p_ee, p_ei, p_ie, p_ii, eps=0.0):
        """create the input weight matrix + the mask of active synapses +
        a mask to block synapses from inh neurons across areas + (bounds of sub_population, number of active synapses in these areas)
        """
        mask, conn_per_bound, bounds = self.make_mask_with_pei(p_ee, p_ei, p_ie, p_ii)
        mask_inter_inh_exc = self.make_mask_inter_inh_exc()
        total_mask = mask * mask_inter_inh_exc
        if self.restrict_inter_area_inh:
            thetas = restricted_inter_inh_thetas_init(
                self.n_units, self.p_exc, self.area_index, eps
            )
        else:
            thetas = torch.rand(self.n_units, self.n_units)
        thetas = thetas * total_mask
        thetas = post_synaptic_weight(thetas, self.p_exc)
        thetas = divide_max_eig(thetas, self.p_exc)

        return thetas, total_mask, mask_inter_inh_exc, (bounds, conn_per_bound)

    def sample_trial_noise(self, batch_size):
        device = self.v_rest.device
        self.trial_noise = torch.rand(batch_size, self.latent_space, device=device)
        self.trial_noise -= 0.5

    def sample_mem_noise(self, input_time, batch_size):
        device = self.v_rest.device
        self.mem_noise = torch.randn(
            input_time, batch_size, self.n_units, device=device
        )

    def reform_biases(self, noise_bias_thr=1.2):
        v_rest = self.v_rest.clone()
        thr = self.base_thr.max()
        thr_off_thr = thr
        v_rest[v_rest > thr_off_thr] = thr_off_thr

        trial_offset = self.trial_offset.clone()
        trial_offset[trial_offset > thr_off_thr] = thr_off_thr
        self.trial_offset.data = trial_offset
        total_sum = trial_offset.sum(0) * 0.5 * self.dt**0.5 + v_rest
        corr = total_sum - self.base_thr
        v_rest[corr > 0] -= corr[corr > 0]
        self.v_rest.data = v_rest

        noise_bias = self.bias.clone()
        noise_bias[noise_bias > noise_bias_thr] = noise_bias_thr
        self.bias.data = noise_bias

    def reform_w_in(self, lr, l1_norm_mult=0.01):
        w_in = self._w_in.clone() - l1_norm_mult * lr
        w_in[w_in < 0] = 1e-12
        self._w_in.data = w_in

    def reform_w_jaw(self, lr, l1_norm_mult=0.01):
        w_jaw = self._w_jaw.clone() - l1_norm_mult * lr
        w_jaw[w_jaw < 0] = 1e-12
        self._w_jaw.data = w_jaw

    def reform_w_rec(
        self, w_rec_orig, mask_rec_orig, lr, l1_norm_mult=0.01, extra_mask=None
    ):
        # mask_rec_orig if 1 don't change if 0 might get triggered to change
        # maybe we add here a noise term
        w_rec_orig.data = w_rec_orig.clone() - l1_norm_mult * lr
        # this is to remove inter inh->exc synapses
        w_rec_orig.data[~(self.mask_inter_inh_exc > 0)] = (
            w_rec_orig.data[~(self.mask_inter_inh_exc > 0)] - lr
        )
        w_rec = w_rec_orig.clone() * mask_rec_orig  # keep the previous active ones
        if extra_mask != None:
            w_rec *= 1 - extra_mask  # keep the previous active ones
        switched = w_rec < 0  # find synapses that switched
        w_rec[switched] = 0  # make them retracted
        mask_rec_orig[switched] = 0  # and also change the mask
        # use this trick so the diagonal is never updated
        mask_rec_orig += self.mask_self_exc
        mask_rec_orig += 1 - self.mask_inter_inh_exc
        if extra_mask != None:
            mask_rec_orig += extra_mask  # here we can add one more mask of neurons that we don't want to touch

        dormant_mask = torch.zeros_like(mask_rec_orig).clone()
        random_mat = torch.ones_like(w_rec) * 1e-12
        for i, ((istart, jstart), (iend, jend)) in enumerate(self.bounds_rec[0]):
            count_sub = (w_rec[istart:iend, jstart:jend] != 0).sum().float()
            # if true there are less than K dormant, here K is symbolic
            if not torch.eq(count_sub, self.bounds_rec[1][i]):
                to_awake = (self.bounds_rec[1][i] - count_sub).int()
                to_turn_cand = torch.where(
                    torch.logical_not(mask_rec_orig[istart:iend, jstart:jend])
                )
                shuffle_cand = (
                    torch.randperm(to_turn_cand[0].shape[0])
                    .long()[:to_awake]
                    .to(w_rec.device)
                )
                xindices = istart + to_turn_cand[0][shuffle_cand]
                yindices = jstart + to_turn_cand[1][shuffle_cand]
                dormant_mask[xindices, yindices] = 1

        if extra_mask != None:
            mask_rec_orig -= extra_mask
        # use this trick so the diagonal is never updated, set back to normal
        mask_rec_orig -= self.mask_self_exc
        mask_rec_orig -= 1 - self.mask_inter_inh_exc
        mask_rec_orig += dormant_mask.clone()
        return w_rec + (dormant_mask * random_mat), mask_rec_orig

    def spike_function(self, x):
        z_forward = (x > 0).float()

        z_backward = torch.where(
            x > 0, -0.5 * torch.square(x - 1), 0.5 * torch.square(x + 1)
        )
        z_backward = torch.where(torch.abs(x) < 1, z_backward, torch.zeros_like(x))
        z_backward = self.dampening_factor * z_backward

        z = (z_forward - z_backward).detach() + z_backward
        return z

    def spike_function_bernoulli(self, x):
        temp = self.temperature
        logit = temp * x
        z_backward = torch.sigmoid(logit) * self.dt / 2
        z_forward = (torch.rand_like(logit) < z_backward).float()
        # z_forward = torch.bernoulli(z_backward)
        z_backward = z_backward * self.dampening_factor
        z = (z_forward - z_backward).detach() + z_backward
        return z

    def prepare_currents(self, spike_buffer, rand_syn_trans=1):
        """Where we calculate the reccurent input currents d

        Args:
            spike_buffer (torch.tensor): past spikes that affect the current timesteps
            rand_syn_trans (float, optional): Only if have random transmission. Defaults to 1.
        """
        w_rec = self._w_rec * rand_syn_trans
        buffer = []
        for g in range(self.rec_groups):
            i, k = torch.where(self.mask[..., g].T)
            buf = spike_buffer[k, :, i].reshape(self.n_units, self.n_delay, -1)
            buf = buf.permute(1, 2, 0)
            buffer.append(buf)
        buffer = torch.stack(buffer)
        exc_buffer = buffer[..., : self.excitatory]
        inh_buffer = buffer[..., self.excitatory :]
        exc_w_rec = w_rec[:, :, : self.excitatory]
        inh_w_rec = w_rec[:, :, self.excitatory :]
        rec_cur_exc = torch.einsum("gtbi,gji->tbj", exc_buffer, exc_w_rec)
        rec_cur_inh = torch.einsum("gtbi,gji->tbj", inh_buffer, inh_w_rec)
        return rec_cur_exc, rec_cur_inh

    def refractoriness(self, z, ref):
        if self.n_refractory >= 1:
            is_ref = ref > 0
            z = torch.where(is_ref, torch.zeros_like(z), z)
            new_ref = torch.where(ref > 0, ref - 1, ref)
            new_ref = torch.where(
                z > 0,
                torch.tensor(self.n_refractory).to(z.device),
                new_ref,
            )
        else:
            new_ref = ref
        return new_ref, z

    def forward(
        self,
        input,
        state,
        light=None,
        sample_mem_noise=True,
        sample_trial_noise=True,
        seed=None,
    ):
        spike_buffer, v, ref, b, jaw_buffer = state

        if light is None:
            dev = input.device
            light = torch.zeros(
                input.shape[0], input.shape[1], self.n_units, device=dev
            )
        light = light * self.light_neuron

        # sample for the membrane and trial noise
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

        if self.trial_offset_bound:
            s = (self.thr0 - self.v_rest0) * 0.2
            self.trial_noise_ready[self.trial_noise_ready.abs() > s] = (
                s * self.trial_noise_ready[self.trial_noise_ready.abs() > s].sign()
            )

        exc_in = self.n_exc_inp
        inp_cur_exc = torch.einsum(
            "tbi,ji->tbj", input[..., :exc_in], self._w_in[:, :exc_in]
        )
        inp_cur_inh = torch.einsum(
            "tbi,ji->tbj", input[..., exc_in:], self._w_in[..., exc_in:]
        )
        inp_cur_exc = inp_cur_exc + light
        mem_noise = self.mem_noise.clone() * self.bias
        if seed is not None:
            torch.manual_seed(seed)
        spike_list = []
        voltage_list = []
        jaw_list = []
        z = spike_buffer[-1]
        self.thr = self.thr0 - self.v_rest
        self.thr_rest_diff = self.thr0 - self.v_rest
        self.thr_rest_diff0 = self.thr0 - self.v_rest
        self.E_exc = self.v_rest0 + (self.thr_rest_diff0) * 2
        self.E_inh = self.v_rest0 - (self.thr_rest_diff0)
        for i in range(input.shape[0] // self.n_delay):
            rec_cur_exc, rec_cur_inh = self.prepare_currents(spike_buffer)
            if self.motor_areas.shape[0] > 0:
                active_buffer = jaw_buffer[
                    : self.jaw_kernel + self.n_delay - 1
                ].permute(1, 2, 0)
                if not self.jaw_open_loop:
                    rec_cur_jaw = self.conv(active_buffer).permute(2, 0, 1)
            for t_idx in range(self.n_delay):
                abs_t_idx = i * self.n_delay + t_idx
                rec_cur = [rec_cur_exc[t_idx], rec_cur_inh[t_idx]]
                if self.motor_areas.shape[0] > 0 and not self.jaw_open_loop:
                    rec_cur.append(rec_cur_jaw[t_idx])
                inp_cur = [inp_cur_exc[abs_t_idx], inp_cur_inh[abs_t_idx]]
                z, v, ref, b, logits = self.step(
                    rec_cur, inp_cur, z, v, ref, b, mem_noise[abs_t_idx]
                )
                jaw_buffer = self.step_jaw(z, jaw_buffer)
                jaw_list.append(jaw_buffer[-1])
                voltage_list.append(logits[0])
                spike_list.append(z[0])
                spike_buffer = torch.cat([spike_buffer[1:], z])
        state = spike_buffer, v, ref, b, jaw_buffer
        jaw = torch.stack(jaw_list)
        voltages = torch.stack(voltage_list)
        spikes = torch.stack(spike_list)
        return spikes, voltages, jaw, state

    def step(self, rec_cur, inp_cur, z, v, ref, b, mem_noise):
        b = self.decay_b * b + (1 - self.decay_b) * z
        thr = self.thr0 + b * self.beta
        # don't allw v_rest to go above threshold
        if (thr <= self.v_rest).any():
            thr[thr <= self.v_rest] += 0.001
        if self.conductance_based:
            exc_cur = (
                (self.E_exc - v)
                * (inp_cur[0] + rec_cur[0])
                / (self.E_exc - self.v_rest0)
            )
            inh_cur = (
                (self.E_inh - v)
                * (inp_cur[1] + rec_cur[1])
                / -(self.E_inh - self.v_rest0)
            )
        else:
            exc_cur = inp_cur[0] + rec_cur[0]
            inh_cur = -(inp_cur[1] + rec_cur[1])
        if torch.isnan(v).sum() > 0:
            print("Network exploded")
        currents = (exc_cur + inh_cur) / 2
        if self.motor_areas.shape[0] > 0 and not self.jaw_open_loop:
            currents += rec_cur[2] / 2
        if self.train_noise_bias:
            mem_noise *= self.thr_rest_diff0 * self._sigma_mem_noise * self.dt**0.5
        else:
            mem_noise *= self.thr_rest_diff * self._sigma_mem_noise * self.dt**0.5
        v = (
            self.decay_v * v
            + (1 - self.decay_v) * (currents + self.trial_noise_ready + self.v_rest)
            + mem_noise
        )
        # Spike generation
        z = self.spike_fun((v - thr) / (thr - self.v_rest0))
        logits = self.temperature * (v.clone() - thr) / (thr - self.v_rest0)
        if not self.spike_fun_type == "sigmoid":
            logits = torch.where(ref > 0, torch.ones_like(logits) * -10, logits)
            ref, z = self.refractoriness(z, ref)
            v -= (thr - self.v_rest) * z
        return z, v, ref, b, logits

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
            size=[self.inter_delay, batch_size, self.n_units], device=device
        )
        voltage0 = (
            torch.zeros(size=[1, batch_size, self.n_units], device=device) + self.v_rest
        )
        ref0 = torch.zeros(
            size=[1, batch_size, self.n_units], device=device, dtype=torch.long
        )
        b0 = torch.zeros(size=[1, batch_size, self.n_units], device=device)
        jaw_buffer0 = torch.zeros(
            size=[self.jaw_delay + self.n_delay - 1, batch_size, 1], device=device
        )
        state0 = (spikes0, voltage0, ref0, b0, jaw_buffer0)
        return state0

    def reform_recurent(self, lr, l1_decay=0.01):
        # here Deep-Rewiring is implemented
        for g in range(self.rec_groups):
            weight, mask = self.reform_w_rec(
                self._w_rec[g].data.clone(),
                self.mask_rec[g].data.clone(),
                lr,
                l1_norm_mult=l1_decay,
            )
            self._w_rec[g].data.copy_(weight.data)
            self.mask_rec[g].data.copy_(mask.data)
            self.mask_rec.data.copy_(mask.data)
        self.reform_w_in(lr, l1_norm_mult=l1_decay)
        self.beta[self.beta < 0] = 0

    def reform_v_rest(self, v_rest_bound=False):
        v = self.v_rest.data
        v0 = self.v_rest0
        t = self.thr0
        # don't allow v_rest to be far away (20% of (t-v0)) from the initial value (v0)
        if v_rest_bound:
            self.v_rest.data[v > v0 + (t - v0) * 0.2] = v0 + (t - v0) * 0.2
            self.v_rest.data[v < v0 - (t - v0) * 0.2] = v0 - (t - v0) * 0.2
        else:
            self.v_rest.data[v > t] = self.thr0
            self.v_rest.data[self.v_rest < 2 * self.v_rest0 - self.thr0] = (
                2 * self.v_rest0 - self.thr0
            )
