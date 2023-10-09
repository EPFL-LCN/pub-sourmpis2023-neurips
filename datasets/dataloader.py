import os
import torch
import numpy as np
import scipy.signal as sig
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedShuffleSplit


class TrialDataset:
    def __init__(
        self,
        root: str,
        sessions,
        areas,
        neurons,
        trial_type=[0, 1, 2, 3],
        stim=[0, 1, 2, 3, 4],
        reaction_time_limits=None,
        start=-1,
        stop=1,
        trial_onset=1,
        with_behaviour=False,
        timestep=0.001,
        random_seed=1,
        start_pre=0.3,
        train_perc=0.75,
    ):
        """Generate the pytorch dataset class for our data. The class loads the data we require based on the
        recordings to model mapping, the timewindow of interest, and the trial type specification.

        Args:
            root (str): Where are the data?
            sessions (np.array): numpy array of strings of session name per neuron
            areas (np.array): numpy array of strings of the area name per neuron
            neurons (np.array): numpy array of ints of the original cluster_index per neuron
            trial_type (list, optional): Which trial types to include, 0:Miss, 1:Hit, 2:CR, 3:FA. Defaults to [0, 1, 2, 3].
            stim (list, optional): Which stimuli to include. Defaults to [0, 1, 2, 3, 4].
            reaction_time_limits ({list, float}, optional): Include Hit trials that respond within a reaction_time_limits window. Defaults to None.
            start (float, optional): start of trial based on trial_onset. Defaults to -1.
            stop (float, optional): stop of trial based on trial_onset. Defaults to 1.
            trial_onset (float, optional): TODO. Defaults to 1.
            with_behaviour (bool, optional): If with_behaviour load also behaviour traces. Defaults to False.
            timestep (float, optional): Timestep. Defaults to 0.001.
            random_seed (int, optional): Random seed. Defaults to 1.
            start_pre (float, optional): We make a small 2nd dataset before start time in order to run the network freely for initialization. Defaults to 0.3.
            train_perc (float, optional): The percentage of train trials. For evaluation, we split the trials in train and test. Defaults to 0.75.
        """
        self.path_ = root
        self.reaction_time_limits = reaction_time_limits
        self.timestep = timestep
        self.trial_onset = trial_onset
        assert start < 1 or start > -1, "start time out of limits"
        assert stop < 1 or stop > -1, "stop time out of limits"
        assert start < stop, "stop should be bigger than start"
        self.start = int(np.round((start + self.trial_onset) / self.timestep))
        self.stop = int(np.round((stop + self.trial_onset) / self.timestep))
        self.relative_start = start
        self.relative_stop = stop
        # the start_pre is the time that we let the network to find a steady state
        self.start_pre = int(
            max(0, (start - start_pre + self.trial_onset)) / self.timestep
        )
        self.with_behaviour = with_behaviour
        self.sessions = sessions
        self.areas = areas
        self.neurons = neurons
        self.trial_type = trial_type
        self.stim = stim
        self.seed = random_seed
        self.train_perc = train_perc
        self.load_data()

    def load_data(self):
        """Load all the data from the ./datasets folder to RAM in a dict"""
        sessions = np.unique(self.sessions)
        sessions = sessions[sessions != ""]
        num_sessions = len(sessions)
        self.data_dict = {
            "sessions": [[] for _ in range(num_sessions)],
            "areas": [[] for _ in range(num_sessions)],
            "neurons": [[] for _ in range(num_sessions)],
            "trial_types": [[] for _ in range(num_sessions)],
            "trial_active": [[] for _ in range(num_sessions)],
            "stims": [[] for _ in range(num_sessions)],
            "reaction_time": [[] for _ in range(num_sessions)],
            "spikes": [[] for _ in range(num_sessions)],
            "spikes_pre": [[] for _ in range(num_sessions)],
            "behaviour": [[] for _ in range(num_sessions)],
            "behaviour_pre": [[] for _ in range(num_sessions)],
            "time": [[] for _ in range(num_sessions)],
            "train_ind": [[] for _ in range(num_sessions)],
            "test_ind": [[] for _ in range(num_sessions)],
        }
        for sess, session in enumerate(sessions):
            self.data_dict["sessions"][sess] = session
            self.load_one_session(sess, session)
            # It guarantees that all trials are equally distributed in the training and testing set
            np.random.seed(self.seed)
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=(1 - self.train_perc), random_state=self.seed
            )
            trial_types = self.data_dict["trial_types"][sess]
            x, y = next(iter(sss.split(np.arange(trial_types.shape[0]), trial_types)))
            self.data_dict["train_ind"][sess] = x
            self.data_dict["test_ind"][sess] = y
        self.data_dict["sessions"] = np.array(self.data_dict["sessions"])

    def load_one_session(self, num_session, session):
        """Loads one session of data from the folder ./datasets to RAM in self.data_dict"""
        # translate trial type number to trial types and reverse
        trial_type_dict = {0: "Miss", 1: "Hit", 2: "CR", 3: "FA"}
        rev_tr_type_dict = {"Miss": 0, "Hit": 1, "CR": 2, "FA": 3}
        trial_type = [trial_type_dict[i] for i in self.trial_type]
        neurons = self.sessions == session
        actual_neurons = self.neurons[neurons]
        areas = self.areas[neurons]

        session_path = os.path.join(self.path_, session)
        trial_info = pd.read_csv(os.path.join(session_path, "trial_info"))

        trial_types, stims = [], []
        # keep trials with correct trial type, stimulus type and reaction time within the reaction_time_limits
        trial_info = trial_info[trial_info.trial_type.isin(trial_type)]
        trial_info = trial_info[trial_info.stim.isin(self.stim)]
        if self.reaction_time_limits is not None:
            no_lick_trials = trial_info.trial_type.isin(["Miss", "CR"])
            lick_trials = trial_info.trial_type.isin(["Hit", "FA"])
            reaction_time = trial_info.reaction_time_jaw > self.reaction_time_limits[0]
            trial_info = trial_info[(lick_trials & reaction_time) | no_lick_trials]
            reaction_time = trial_info.reaction_time_jaw < self.reaction_time_limits[1]
            trial_info = trial_info[(lick_trials & reaction_time) | no_lick_trials]

        trial_types = trial_info.trial_type.values
        trial_types = np.array([rev_tr_type_dict[j] for j in trial_types])
        trial_active = trial_info.trial_active.values
        stims = trial_info.stim.values
        trial_onsets = trial_info.trial_onset.values
        reaction_time = trial_info.reaction_time_jaw.values

        # load all spikes of a session
        spike_times, spike_clusters = [], []
        for i, neuron in enumerate(actual_neurons):
            spike_times.append(
                np.load(os.path.join(session_path, f"neuron_index_{int(neuron)}.npy"))
            )
            spike_clusters.append(np.ones_like(spike_times[-1]) * i)
        spike_times = (np.concatenate(spike_times) // self.timestep).astype(int)
        spike_clusters = np.concatenate(spike_clusters).astype(int)
        data = np.ones_like(spike_clusters).astype(int)
        raster = csr_matrix(
            (data, (spike_times, spike_clusters)),
            (spike_times.max() + 1, spike_clusters.max() + 1),
        )
        # init arrays with trials
        time_dur = self.stop - self.start
        time_dur_pre = self.start - self.start_pre
        spikes = np.zeros((trial_info.shape[0], neurons.sum(), time_dur))
        spikes = spikes.astype(np.float32)
        spikes_pre = np.zeros((trial_info.shape[0], neurons.sum(), time_dur_pre))
        spikes_pre = spikes_pre.astype(np.float32)
        if self.with_behaviour:
            behaviour = np.zeros((trial_info.shape[0], 3, time_dur))
            behaviour = behaviour.astype(np.float32)
            behaviour_pre = np.zeros((trial_info.shape[0], 3, time_dur_pre))
            behaviour_pre = behaviour_pre.astype(np.float32)
        else:
            behaviour, behaviour_pre = [], []

        # load data into array with dimensions Trials x Neurons x Time
        whisker_paths = trial_info.whisker_angle.values
        jaw_paths = trial_info.jaw_trace.values
        tongue_paths = trial_info.tongue_trace.values
        trials = []
        for i, trial_onset in enumerate(trial_onsets):
            start = int(np.round((trial_onset) / self.timestep)) + self.start
            start_pre = start - int(np.round(0.3 / self.timestep))
            stop = start + (self.stop - self.start)
            if raster[start:stop].shape[0] < stop - start:
                break
            spikes[i] = raster[start:stop].toarray().T
            spikes_pre[i] = raster[start_pre:start].toarray().T
            if self.with_behaviour:
                dt_orig = 2  # the camera has 500 fps
                dt = int(self.timestep * 1000)  # in ms
                assert dt % 1 == 0, "no implementation for this dt"
                wh = np.load(whisker_paths[i] + ".npy")
                jaw = np.load(jaw_paths[i] + ".npy")
                tongue = np.load(tongue_paths[i] + ".npy")
                wh = sig.resample_poly(wh, dt_orig, dt)
                jaw = sig.resample_poly(jaw, dt_orig, dt)
                tongue = sig.resample_poly(tongue, dt_orig, dt)
                behaviour[i] = np.vstack([wh, jaw, tongue])[:, self.start : self.stop]
                behaviour_pre[i] = np.vstack([wh, jaw, tongue])[
                    :, self.start_pre : self.start
                ]
                if np.isnan(behaviour[i].sum()):
                    continue
            trials.append(i)
        trials = np.array(trials)
        self.data_dict["areas"][num_session] = areas
        self.data_dict["neurons"][num_session] = neurons
        self.data_dict["trial_types"][num_session] = trial_types[trials]
        self.data_dict["trial_active"][num_session] = trial_active[trials]
        self.data_dict["stims"][num_session] = stims[trials]
        self.data_dict["reaction_time"][num_session] = reaction_time[trials]
        self.data_dict["spikes"][num_session] = spikes[trials]
        self.data_dict["spikes_pre"][num_session] = spikes_pre[trials]
        if self.with_behaviour:
            self.data_dict["behaviour"][num_session] = behaviour[trials]
            self.data_dict["behaviour_pre"][num_session] = behaviour_pre[trials]
        else:
            self.data_dict["behaviour"][num_session] = behaviour
            self.data_dict["behaviour_pre"][num_session] = behaviour_pre

    def max_trials(self, train=2, trial_type=[0, 1, 2, 3]):
        """Find the max trials of a session in the training or/and testing set

        Args:
            train (int): If 0 only test set, if 1 only train, if 2 both, Defaults to 2.
            trial_type (list, optional): Trial types to count. Defaults to [0, 1, 2, 3], which are all trials.

        Returns:
            int: number of trials
        """
        # if train = 0 (test set), 1 (train set), 2 (both)
        test_trials = [
            len(np.isin(self.data_dict["trial_types"][i][j], trial_type))
            for i, j in enumerate(self.data_dict["test_ind"])
        ]
        train_trials = [
            len(np.isin(self.data_dict["trial_types"][i][j], trial_type))
            for i, j in enumerate(self.data_dict["train_ind"])
        ]
        if train == 0:
            return max(test_trials)
        elif train == 1:
            return max(train_trials)
        else:
            return max(test_trials) + max(train_trials)

    def to_torch(self, trial_types=[0, 1, 2, 3]):
        """Move to torch the data that where loaded with load_data

        Args:
            trial_types (list, optional): _description_. Defaults to [0, 1, 2, 3].
        """
        n_units = self.neurons.shape[0]
        sessions = self.data_dict["sessions"]
        timefull = self.stop - self.start
        timepre = self.start - self.start_pre
        # for convenience we put the data from all sessions in a big tensor
        # where if data are missing are replaced with nans
        trials = self.max_trials()
        self.spikes_all = torch.ones(timefull, trials, n_units) * torch.nan
        self.spikes_all_pre = torch.ones(timepre, trials, n_units) * torch.nan
        if self.with_behaviour:
            self.jaw = torch.ones(timefull, trials, len(sessions)) * torch.nan
            self.tongue = torch.ones(timefull, trials, len(sessions)) * torch.nan
        # we synchronize the previous tensor with the a list called session_info
        # entry 1. trial_types, 2. stims, 3. active_quiet, 4. neurons
        self.session_info = [[], [], [], []]
        for i, session in enumerate(sessions):
            index = int(np.where(sessions == session)[0])
            indices = np.where(
                np.isin(self.data_dict["trial_types"][index], trial_types)
            )[0]
            spikes = torch.tensor(self.data_dict["spikes"][index][indices])
            spikes = spikes.permute(2, 0, 1)
            trials = spikes.shape[1]
            spikes_pre = torch.tensor(self.data_dict["spikes_pre"][index][indices])
            spikes_pre = spikes_pre.permute(2, 0, 1)
            self.session_info[0].append(self.data_dict["trial_types"][index][indices])
            self.session_info[1].append(self.data_dict["stims"][index][indices])
            self.session_info[2].append(self.data_dict["trial_active"][index][indices])
            self.session_info[3].append(self.data_dict["neurons"][index])

            self.spikes_all[:, :trials, self.session_info[3][-1]] = spikes
            self.spikes_all_pre[:, :trials, self.session_info[3][-1]] = spikes_pre
            if self.with_behaviour:
                self.jaw[:, :trials, i] = torch.tensor(
                    self.data_dict["behaviour"][index][indices, 1].T
                )
                self.tongue[:, :trials, i] = torch.tensor(
                    self.data_dict["behaviour"][index][indices, 2].T
                )

    def normalize_jaw(self):
        """Standard normalization of the jaw based on the baseline activity"""
        start = -int(self.relative_start / self.timestep)
        jaw_mean_base = self.jaw[:start].nanmean((0, 1))
        std = lambda x: (((x**2).nanmean(1) - x.nanmean(1) ** 2)).mean(0) ** 0.5
        jaw_std_base = std(self.jaw)
        self.jaw = -(self.jaw - jaw_mean_base) / jaw_std_base
        tongue_mean_base = self.tongue[:start].nanmean((0, 1))
        tongue_std_base = std(self.tongue)
        self.tongue = -(self.tongue - tongue_mean_base) / tongue_std_base

    def get_train_trial_type(
        self, train=0, trial_type=[0, 1, 2, 3], device="cpu", jaw_tongue=1
    ):
        """After to_torch method we can run this method to isolate the data into training/test set
        or keep only specific trial types, we can also specify to use only the jaw or the tongue

        Args:
            train (int): If 0 only test set, if 1 only train, if 2 both, Defaults to 2.
            trial_type (list, optional): Trial types to isolate. Defaults to [0, 1, 2, 3].
            device (str, optional): Which device to use {"cuda", "cpu"}. Defaults to "cpu".
            jaw_tongue (int, optional): If 1 then it uses the jaw, if 2 the tongue. Defaults to 1.

        Returns:
            [spikes, spikes_pre, jaw, session_info]: _description_
        """
        assert jaw_tongue in [1, 2], "wrong jaw_tongue value"
        sessions = self.data_dict["sessions"]
        timefull, _, n_units = self.spikes_all.shape
        timepre, _, n_units = self.spikes_all_pre.shape
        max_trials = self.max_trials(train, trial_type=trial_type)
        spikes = torch.zeros(timefull, max_trials, n_units) * torch.nan
        spikes_pre = torch.zeros(timepre, max_trials, n_units) * torch.nan
        jaw = None
        if self.with_behaviour:
            jaw = torch.zeros(timefull, max_trials, len(sessions)) * torch.nan
        else:
            jaw = None
        new_session_info = [[] for i in range(len(self.session_info) - 1)]
        new_session_info.append(self.session_info[-1])
        for session in range(len(self.session_info[0])):
            indices_train = np.zeros_like(self.session_info[0][session]) > 0
            if train == 0:
                indices_t = self.data_dict["test_ind"][session]
                indices_train[indices_t] = True
            elif train == 1:
                indices_t = self.data_dict["train_ind"][session]
                indices_train[indices_t] = True
            else:
                indices_train = ~indices_train
            indices_trial_type = np.isin(self.session_info[0][session], trial_type)
            indices = torch.arange(len(self.session_info[0][session]))
            indices = indices[indices_train & indices_trial_type]
            sess_neurons = self.session_info[-1][session]
            trials = len(indices)
            spikes[:, :trials, sess_neurons] = self.spikes_all[:, indices][
                ..., sess_neurons
            ]
            spikes_pre[:, :trials, sess_neurons] = self.spikes_all_pre[:, indices][
                ..., sess_neurons
            ]
            if self.with_behaviour:
                if jaw_tongue == 1:
                    tmp = self.jaw[:, indices][..., session]
                elif jaw_tongue == 2:
                    tmp = self.tongue[:, indices][..., session]
                jaw[:, :trials, session] = tmp - tmp[: self.start].mean(0)
                jaw = jaw.to(device)
            for i in range(len(self.session_info) - 1):
                new_session_info[i].append(self.session_info[i][session][indices])
        return (
            spikes.to(device),
            spikes_pre.to(device),
            jaw,
            new_session_info,
        )


def keep_trial_types_behaviour(
    data_spikes, jaw, rewrite_spikes, rewrite_jaw, session_info, trial_types=[0, 1]
):
    """A fast way to isolate particular

    Args:
        data_spikes (_type_): _description_
        jaw (_type_): _description_
        rewrite_spikes (_type_): _description_
        rewrite_jaw (_type_): _description_
        session_info (_type_): _description_
        trial_types (list, optional): _description_. Defaults to [0, 1].

    Returns:
        (spikes, jaw, session_info): _description_
    """
    keep = []
    max_trials = 0
    for session in range(len(session_info[0])):
        isin = np.isin(session_info[0][session], trial_types)
        keep.append(torch.tensor(np.where(isin)[0]))
        max_trials = max(isin.sum(), max_trials)

    rewrite_spikes *= torch.nan
    if jaw is not None:
        rewrite_jaw *= torch.nan
    session_info_new = [[], [], [], session_info[-1]]
    for session in range(len(session_info[0])):
        for i in range(len(session_info) - 1):
            session_info_new[i].append(session_info[i][session][keep[session]])
        trials = keep[session].shape[0]
        neurons = torch.tensor(session_info_new[-1][session])
        rewrite_spikes[:, :trials, neurons] = data_spikes[:, keep[session]][
            ..., neurons
        ]
        if jaw is not None:
            rewrite_jaw[:, :trials, session] = jaw[:, keep[session]][..., session]

    return rewrite_spikes, rewrite_jaw, session_info_new


def balance_trial_type(data_spikes, data_jaw, session_info, data_perc, seed=None):
    """Resample trials from the data so the new set of trials have a similar trial
    type distribution as the data
    """
    data_spikes_new = torch.zeros_like(data_spikes) * torch.nan
    if data_jaw is not None:
        data_jaw_new = torch.zeros_like(data_jaw) * torch.nan
    else:
        data_jaw_new = None
    session_info_new = [[] for i in range(len(session_info))]
    for id_sess in range(len(session_info[0])):
        sess_perc = (
            np.unique(session_info[0][id_sess], return_counts=True)[1]
            / session_info[0][id_sess].shape[0]
        )
        assert data_perc.shape == sess_perc.shape, "no same classes"

        # which trial is the most underrepresented
        under_class = (data_perc / sess_perc).argmax()
        # which is the first class
        add_to_class = session_info[0][id_sess].min()
        # actual trials that can be utilized if we restrict the balance
        K = (session_info[0][id_sess] == under_class + add_to_class).sum() / data_perc[
            under_class
        ]

        K = K.astype(int)
        # number of each trial type with balanced percentage
        count_balanced = np.round(K * data_perc).astype(int)
        K = count_balanced.sum()
        count = 0
        # new index with which we will reform the data
        idx = np.zeros(count_balanced.sum()).astype(int)
        for i, value in enumerate(range(add_to_class, len(data_perc) + add_to_class)):
            condition_ids = np.where(session_info[0][id_sess] == value)[0]
            np.random.seed(seed)
            np.random.shuffle(condition_ids)
            idx[count : count + count_balanced[i]] = condition_ids[: count_balanced[i]]
            count += count_balanced[i]

        neurons_sess = session_info[-1][id_sess]
        for i in range(len(session_info) - 1):
            session_info_new[i].append(session_info[i][id_sess][idx])
        session_info_new[-1].append(neurons_sess)
        data_spikes_new[:, :K, neurons_sess] = data_spikes[:, idx][
            ..., session_info[-1][id_sess]
        ]
        if data_jaw is not None:
            data_jaw_new[:, :K, id_sess] = data_jaw[:, idx, id_sess]
    return data_spikes_new, data_jaw_new, session_info_new
