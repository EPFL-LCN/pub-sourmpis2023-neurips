import numpy as np
import pandas as pd
import os


def build_network(
    rsnn,
    datapath,
    area_list=["wS1", "wS2", "wM1", "wM2", "ALM", "tjM1"],
    with_video=True,
    hidden_propability=0.0,
):
    """
    Build the network based on the data collected from the dataset and based on the array
    rsnn.area_index and rsnn.excitatory_index that specifies the area  and neurotransmitter type
    of each neuron.
    Sample without replacement from the list of the neurons from the sessions
    that have video behavior when with_video True, else sample from any possible session.
    Also, choose p_exc of the neurons to be excitatory. The excitation is defined as
    Peak to Peak time in the spike template is more than 0.56ms.
    """
    clusterdf = pd.read_csv(os.path.join(datapath, "cluster_information"))
    clusterdfwhole = clusterdf.copy()  # for statistics for hidden neurons
    if with_video:
        clusterdf = clusterdf[clusterdf["with_video"] == with_video]
    clusterdf = clusterdf[clusterdf["area"].isin(area_list)]
    # keep neurons with less than 40 Hz
    clusterdf = clusterdf[clusterdf.firing_rate < 40]

    n_neurons = rsnn.n_units
    # init of the output arrays
    firing_rate = np.zeros(n_neurons)
    neuron_index = np.zeros(n_neurons)
    session = np.zeros(n_neurons).astype(str)
    areas = np.zeros(n_neurons).astype(str)

    # order sessions based on the number of neurons they have
    # this prioritize sessions with more neurons
    sessions_uniq = clusterdf.session.value_counts().index.values
    sess_exc = [sessions_uniq.copy() for area in area_list]
    sess_inh = [sessions_uniq.copy() for area in area_list]
    # in this loop we make the matching from recordings to simulations
    for i in range(n_neurons):
        exc = rsnn.excitatory_index[i].numpy()
        area_id = rsnn.area_index[i]
        area = area_list[area_id]
        cur_session = sess_exc[area_id] if exc else sess_inh[area_id]
        possible_ids = clusterdf[
            (clusterdf["area"] == area)
            & (clusterdf["excitatory"] == exc)
            & (clusterdf["session"] == cur_session[0])
        ]
        # making sure that we use all neurons from a session and then move to the next session
        while possible_ids.shape[0] == 0 and area != "Nope":
            cur_session = cur_session[1:]
            possible_ids = clusterdf[
                (clusterdf["area"] == area)
                & (clusterdf["excitatory"] == exc)
                & (clusterdf["session"] == cur_session[0])
            ]

        if possible_ids.shape[0] == 0 and area != "Nope":
            if hidden_propability == 0:
                assert False, "not enough neurons in the dataset"
            else:
                print("filling with hidden")

        # in the first case of the if statement we check if we add a hidden neuron
        # we add hidden neurons, if we have hidden_probability > 0 or the area is a Nope area
        flag_hidden = np.random.rand() < hidden_propability
        if flag_hidden or area == "Nope":
            neuron_index[i] = -1  # default for hidden neurons
            session[i] = ""
            areas[i] = area
            # if neuron is hidden, use firing rate the firing rate of random neuron from the dataset
            possible_ids = clusterdfwhole[
                (clusterdfwhole["area"] == area) & (clusterdfwhole["excitatory"] == exc)
            ]
            # use the firing rate of a random neuron from same group (area/exc-inh)
            index = np.random.choice(clusterdfwhole.index)
            firing_rate[i] = clusterdfwhole.firing_rate[
                clusterdfwhole.index == index
            ].values
        else:
            # from the possible neurons pick one
            index = np.random.choice(possible_ids.index)
            neuron_index[i] = clusterdf.cluster_index[clusterdf.index == index].values
            firing_rate[i] = clusterdf.firing_rate[clusterdf.index == index].values
            session[i] = clusterdf.session[clusterdf.index == index].values[0]
            areas[i] = clusterdf.area[clusterdf.index == index].values[0]
            clusterdf = clusterdf[clusterdf.index != index]
    return neuron_index, firing_rate, session, areas
