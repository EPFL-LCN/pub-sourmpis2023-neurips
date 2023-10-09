import torch
import torch.nn as nn
import numpy as np


class InputSpikes(nn.Module):
    def __init__(self, opt, seeded=False, prec_type=torch.float32):
        super(InputSpikes, self).__init__()
        self.prec_type = prec_type
        self.opt = opt
        self.seeded = seeded
        self.n_stim_amp = opt.n_stim_amp + 1
        if opt.n_rnn_in == 200:
            p_stim = 0.5
        elif opt.n_rnn_in == 300:
            p_stim = 2 / 3
        else:
            p_stim = 1
        self.p_stim = p_stim

    def forward(self, stim):
        dt_in_seconds = self.opt.dt / 1000.0
        n_neurons = self.opt.n_rnn_in
        start = -int(self.opt.start / dt_in_seconds)
        stop = int(self.opt.stop / dt_in_seconds)
        thalamic_delay = int(self.opt.thalamic_delay / dt_in_seconds)
        stim_dur = int(self.opt.stim_duration / dt_in_seconds)
        firing_prob = self.opt.input_f0 * dt_in_seconds
        batch_size = stim.shape[0]
        pattern = (
            torch.ones((start + stop, batch_size, n_neurons), device=stim.device)
            * firing_prob
        )
        pattern[pattern > 1] = 1
        n_stim = int(n_neurons * self.p_stim / len(self.opt.stim_onsets))
        for i, j in enumerate(stim):
            for l, k in enumerate(self.opt.stim_onsets):
                start = int(np.round((k - self.opt.start) / dt_in_seconds))
                scale = (
                    self.opt.scale_fun(j.item()) if k == 0 else self.opt.scale_fun(1)
                )
                pattern[
                    start + thalamic_delay : start + thalamic_delay + stim_dur,
                    i,
                    n_stim * l : n_stim * (l + 1),
                ] *= (
                    1 + self.opt.stim_valance * scale
                )
        pattern[pattern > 1] = 1
        if n_neurons == 2:
            return (pattern > firing_prob) * 1.0
        return torch.bernoulli(pattern)
