import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


def make_onset_timeseries(time, length):
    x = np.zeros(length)
    x[time] = 1
    return x


def make_kernels(tau_rise, tau_fall, timestep):
    """Make the kernel to convolve the onset timeseries."""
    max_time = int(8 * tau_fall / timestep)
    time = np.arange(max_time) * timestep
    kernel = lambda x: (np.exp(-x / tau_fall) - np.exp(-x / tau_rise))
    return kernel(time)


def firing_rates(num_neurons, average, std, mode="log_normal"):
    if mode == "log_normal":
        distro = np.random.randn(num_neurons) * np.log(std) + np.log(average)
        distro = np.exp(distro)
        # we don't allow extreme firing rates
        distro[distro > 40] = 40 - np.random.rand((distro > 40).sum()) * 10
    elif mode == "normal":
        distro = np.random.randn(num_neurons) * std + average
    elif mode == "none":
        distro = np.ones(num_neurons) * average
    return distro


class PseudoData:
    def __init__(
        self,
        path,
        mod_strength=[-2, -1, 0, 1, 2],
        mod_prob=[0.05, 0.05, 0.2, 0.5, 0.2],
        variation=100,
        num_sessions=20,
        trials_per_session=100,
        neurons_per_sess=100,
        pEI=0.8,
        length=4000,
        trial_onset=1000,
        onsets=[5, 40],
        tau_rise=5,
        tau_fall=20,
        timestep=1,
        firing_rates=[3, 4],
        firing_rates_std=[2, 2],
        p_trial_type=0.8,
    ):
        """Generate a dataset with artificial data. The main idea is that we have a dataset
        of two areas where the first is responding always to a stimulus input at onsets[0]
        and the sencond area responds with p_trial_type probability at onsets[1].

        Args:
            path (_type_): _description_
            mod_strength (list, optional): strength of modulation. Defaults to [-2, -1, 0, 1, 2].
            mod_prob (list, optional): mod_prob[i] neurons have are modulated to the stimulus with strength mod_strength[i]. Defaults to [0.05, 0.05, 0.2, 0.5, 0.2].
            variation (int, optional): variation of the onset of the second are in timesteps. Defaults to 100.
            num_sessions (int, optional): Number of sesssions to generate. Defaults to 20.
            trials_per_session (int, optional): Number of trials per session. Defaults to 100.
            neurons_per_sess (int, optional): Number of neurons per session. Defaults to 100.
            pEI (float, optional): Ratio of excitatory neuron. Defaults to 0.8.
            length (int, optional): Trial timesteps. Defaults to 4000.
            trial_onset (int, optional): Onset of the trial. Defaults to 1000.
            onsets (list, optional): onsets[i] is the onset of the stimulus in areas[i] relative to the trial_onset. Defaults to [5, 40].
            tau_rise (int, optional): rise timeconstant of the stimulus kernel. Defaults to 5.
            tau_fall (int, optional): fall timeconstant of the stimulus kernel. Defaults to 20.
            timestep (int, optional): Timestep length usually 1ms.
            firing_rates (list, optional): firing_rates[0] = mean firing rate of excitatory neurons. Defaults to [3, 4].
            firing_rates_std (list, optional): firing_rates_std[0] = mean firing rate of excitatory neurons. Defaults to [2, 2].
            p_trial_type (float, optional): probability of a hit trial(both areas active). Defaults to 0.8.
        """
        self.path = path
        self.mod_strength = mod_strength
        self.mod_prob = mod_prob
        self.variation = variation
        self.num_sessions = num_sessions
        self.trials_per_session = trials_per_session
        self.neurons_per_sess = neurons_per_sess
        self.pEI = pEI
        self.length = length
        self.trial_onset = trial_onset
        self.onsets = onsets
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall
        self.timestep = timestep
        self.firing_rates = firing_rates
        self.firing_rates_std = firing_rates_std
        self.p_tt = p_trial_type
        if os.path.exists(self.path):
            print("path exists rewriting")
        else:
            os.mkdir(self.path)
        print(f"Generating dataset {self.path.split('/')[-1]}")
        print("Generating cluster info")
        self.make_cluster_info()
        print("Generating trial info")
        self.make_trial_info()
        print("Generating spikes")
        self.make_spikes()

    def make_cluster_info(self):
        """Generate neuron profile"""
        all_cluster_df = pd.DataFrame(
            columns=(
                "session",
                "area",
                "excitatory",
                "firing_rate",
                "cluster_index",
            )
        )
        total_neurons = self.num_sessions * self.neurons_per_sess
        total_neurons_exc = int(total_neurons * self.pEI)
        total_neurons_inh = total_neurons - total_neurons_exc
        fr_exc = firing_rates(
            total_neurons_exc,
            self.firing_rates[0],
            self.firing_rates_std[0],
        )
        fr_inh = firing_rates(
            total_neurons_inh,
            self.firing_rates[1],
            self.firing_rates_std[0],
        )

        e_ind, i_ind = 0, 0
        # a session can have neurons from either both areas or from one of the two
        areas_rot = [["area1", "area2"], ["area1"], ["area2"]]
        areas = ["area1", "area2"]
        for sess in range(self.num_sessions):
            sess_cluster_df = pd.DataFrame(
                columns=(
                    "neuron_index",
                    "area",
                    "excitatory",
                    "firing_rate",
                )
            )
            # where PS stands for PseudoSession
            session_path = "PS{:03d}_20220404".format(sess)
            if not os.path.exists(os.path.join(self.path, session_path)):
                os.mkdir(os.path.join(self.path, session_path))
            os.mkdir(os.path.join(self.path, session_path, areas[0]))
            if len(areas) > 1:
                os.mkdir(os.path.join(self.path, session_path, areas[1]))
            # generate one table per session and a global one (the second helps to construct the RSNN)
            sess_entry = pd.DataFrame(columns=sess_cluster_df.columns)
            all_entry = pd.DataFrame(columns=all_cluster_df.columns)
            areas = areas_rot[sess % 3]
            for n in range(self.neurons_per_sess):
                exc = int(self.pEI * self.neurons_per_sess)
                inh = self.neurons_per_sess - exc
                area = (n > exc) * (n - exc) // (inh // len(areas)) + (n < exc) * n // (
                    exc // len(areas)
                )
                excitatory = n < exc
                sess_entry["neuron_index"] = [n]
                sess_entry["area"] = [areas[area]]
                sess_entry["excitatory"] = [excitatory]
                sess_entry["firing_rate"] = [
                    (fr_exc[e_ind] if excitatory else fr_inh[i_ind])
                ]
                all_entry["session"] = [session_path]
                all_entry["area"] = [areas[area]]
                all_entry["excitatory"] = [excitatory]
                all_entry["firing_rate"] = [
                    (fr_exc[e_ind] if excitatory else fr_inh[i_ind])
                ]
                all_entry["cluster_index"] = [n]
                e_ind += excitatory
                i_ind += not excitatory

                sess_cluster_df = pd.concat(
                    [sess_cluster_df, sess_entry], ignore_index=True
                )
                all_cluster_df = pd.concat(
                    [all_cluster_df, all_entry], ignore_index=True
                )
            sess_cluster_df.to_csv(
                os.path.join(self.path, all_entry["session"][0], "cluster_info")
            )
        all_cluster_df.to_csv(os.path.join(self.path, "cluster_information"))

    def make_trial_info(self):
        """Make the csv of trial info, the trial structure is the same as the DataFromVahid_expert"""

        for sess in range(self.num_sessions):
            session_path = "PS{:03d}_20220404".format(sess)
            trial_info_df = pd.DataFrame(
                columns=(
                    "trial_number",
                    "reaction_time_jaw",
                    "reaction_time_piezo",
                    "stim",
                    "trial_type",
                    "trial_active",
                    "trial_onset",
                    "jaw_trace",
                    "tongue_trace",
                    "whisker_angle",
                    "completed_trials",
                    "video_onset",
                    "video_offset",
                )
            )
            pseudo_signal_path = os.path.join(self.path, session_path, "pseudo_signal")
            os.mkdir(pseudo_signal_path)
            entry_df = pd.DataFrame(columns=trial_info_df.columns)
            for trial in range(self.trials_per_session):
                entry_df["trial_number"] = [trial]
                # the next seven keys are legacy values and not really used
                entry_df["reaction_time_jaw"] = [0.15]
                entry_df["reaction_time_piezo"] = [0.15]
                entry_df["stim"] = [4]
                entry_df["trial_active"] = [np.random.choice([0, 1], p=[0.5, 0.5])]
                entry_df["video_onset"] = [2]
                entry_df["video_offset"] = [0]
                #
                entry_df["trial_type"] = [
                    np.random.choice(["Miss", "Hit"], p=[1 - self.p_tt, self.p_tt])
                ]
                entry_df["trial_onset"] = [trial * int(self.length / 1000)]
                trial_info_df = pd.concat([trial_info_df, entry_df], ignore_index=True)
            trial_info_df.to_csv(os.path.join(self.path, session_path, "trial_info"))

    def make_spikes(self):
        """Based on the cluster_info and trial_info generate the appropriate neuronal activity for all the neurons and trials."""

        sessions = os.listdir(self.path)
        sessions = [sess for sess in sessions if sess != "cluster_information"]
        clusters = pd.read_csv(os.path.join(self.path, "cluster_information"))
        for i in tqdm(range(len(sessions)), desc="session"):
            sess = sessions[i]
            trial_info = pd.read_csv(os.path.join(self.path, sess, "trial_info"))
            trial_types = trial_info.trial_type.values
            session_path = os.path.join(self.path, sess)
            cluster = clusters[clusters.session == sess]
            areas = cluster.area.values
            area1, area2 = self.trial_prototypes()
            # area2 responds only in Hit trials
            area2 = (area2.T * (trial_types == "Hit")).T
            # neurons can be positive, negative or no modulated by the stimulus
            spike_times = [[] for _ in range(cluster.shape[0])]
            modulation_area2 = np.random.choice(
                self.mod_strength, cluster.shape[0], p=self.mod_prob
            )
            modulation_area1 = np.random.choice(
                self.mod_strength, cluster.shape[0], p=self.mod_prob
            )
            time = 0
            for trial in range(self.trials_per_session):
                area2_t = area2[trial, :, None] @ modulation_area2[None] + 1
                area1_t = area1[trial, :, None] @ modulation_area1[None] + 1
                area1_t[area1_t < 0] = 0
                rates = area1_t * (areas == "area1") + area2_t * (areas == "area2")
                rates *= cluster.firing_rate.values
                threshold = np.random.rand(rates.shape[0], rates.shape[1])
                spikes = (rates * self.timestep / 1000) > threshold
                spike_tms, spike_neurons = np.where(spikes)
                for n in range(len(spike_times)):
                    spike_times[n] += (
                        spike_tms[spike_neurons == n] / 1000 + time
                    ).tolist()
                time += self.length / 1000

            for cl in range(cluster.shape[0]):
                np.save(
                    os.path.join(session_path, f"neuron_index_{cl}"), spike_times[cl]
                )

    def trial_prototypes(self):
        """Generate the population average activity of every area for every trial."""
        # all times are in ms
        sig = make_onset_timeseries(self.onsets[0] + self.trial_onset, self.length)
        kernel1 = make_kernels(self.tau_rise, self.tau_fall, self.timestep)
        sig1 = np.convolve(sig, kernel1)[: -kernel1.shape[0] + 1]
        sig1 /= sig1.max()
        sig1 *= 3
        sigs_area1 = []
        sigs_area2 = []
        for _ in range(self.trials_per_session):
            sigs_area1.append(sig1)
            onset_area2 = self.trial_onset + self.onsets[1]
            onset_area2 += np.random.randint(0, self.variation)
            sig = make_onset_timeseries(onset_area2, self.length)
            kernel = make_kernels(self.tau_rise, self.tau_fall, self.timestep)
            sig = np.convolve(sig, kernel)[: -kernel.shape[0] + 1]
            sig /= sig.max()
            sig *= 3
            sigs_area2.append(sig)

        sigs_area1 = np.stack(sigs_area1)
        sig_area2 = np.stack(sigs_area2)
        return sigs_area1, sig_area2


if __name__ == "__main__":
    # Dataset for Figure 2 and supplementary A2
    conf = {
        "onsets": [4, 8],
        "tau_rise": 5,
        "tau_fall": 20,
        "firing_rates": [8, 16],
        "firing_rates_std": [2, 1],
        "p_trial_type": 0.8,
        "trials_per_session": 200,
        "mod_prob": [0, 0, 0.2, 0.6, 0.2],
        "pEI": 0.8,
        "variation": 1,
    }
    for i in [8]:
        conf["onsets"][1] = i
        path = f"datasets/PseudoData_v16_variation1ms"
        pseudo_data = PseudoData(
            path,
            variation=conf["variation"],
            onsets=conf["onsets"],
            tau_rise=conf["tau_rise"],
            tau_fall=conf["tau_fall"],
            firing_rates=conf["firing_rates"],
            firing_rates_std=conf["firing_rates_std"],
            p_trial_type=conf["p_trial_type"],
            trials_per_session=conf["trials_per_session"],
            mod_prob=conf["mod_prob"],
            pEI=conf["pEI"],
            num_sessions=10,
            neurons_per_sess=100,
        )
        json.dump(conf, open(path + "/conf.json", "w"))

    # Dataset for supplementary A5
    conf = {
        "onsets": [4, 8],
        "tau_rise": 5,
        "tau_fall": 20,
        "firing_rates": [8, 16],
        "firing_rates_std": [2, 1],
        "p_trial_type": 0.8,
        "trials_per_session": 200,
        "mod_prob": [0, 0, 0.2, 0.6, 0.2],
        "pEI": 0.8,
        "variation": 1,
    }
    for i in [20, 200]:
        conf["onsets"][1] = i
        path = f"datasets/PseudoData_v17_delay{conf['onsets'][1]}_onesession"
        pseudo_data = PseudoData(
            path,
            variation=conf["variation"],
            onsets=conf["onsets"],
            tau_rise=conf["tau_rise"],
            tau_fall=conf["tau_fall"],
            firing_rates=conf["firing_rates"],
            firing_rates_std=conf["firing_rates_std"],
            p_trial_type=conf["p_trial_type"],
            trials_per_session=conf["trials_per_session"],
            mod_prob=conf["mod_prob"],
            pEI=conf["pEI"],
            num_sessions=1,
            neurons_per_sess=500,
        )
        json.dump(conf, open(path + "/conf.json", "w"))
