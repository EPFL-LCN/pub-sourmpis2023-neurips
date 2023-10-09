import numpy as np
from mat73 import loadmat
import os
import pandas as pd


def save_datastructure(
    path="datasets/DataFromVahid",
    expert=True,
    save_video=True,
    save_clusters_info=True,
):
    """Transforms the matlab file with the data in a folder structure that can be loaded easily to a pytorch dataset.


    Args:
        path (str, optional): Path to save the resulting directory. Defaults to "datasets/DataFromVahid".
        expert (bool, optional): Save the expert or novice animal. For the paper, it's only the expert. Defaults to True.
        save_video (bool, optional): If to save the video traces of the jaw and the whisker. Defaults to True.
        save_clusters_info (bool, optional): If to save the neuron data. Defaults to True.
    """
    if expert:
        path = path + "_expert"
    else:
        path = path + "_novice"
    mat = loadmat("datasets/spikeData_v9.mat")["spikeData_v9"]
    if not os.path.exists(path):
        os.mkdir(path)
    for i, sess in enumerate(mat["mouse"]):
        if mat["expert"][i] != expert:
            continue
        session_path = os.path.join(path, sess + "_" + mat["date"][i])
        try:
            os.mkdir(session_path)
        except:
            print("directory already exists we override")
        trial_df = pd.DataFrame(
            columns=(
                "trial_number",
                "reaction_time_piezo",
                "reaction_time_jaw",
                "stim",
                "trial_active",
                "trial_type",
                "trial_onset",
                "jaw_trace",
                "tongue_trace",
                "whisker_angle",
                "completed_trials",
                "video_onset",
                "video_offset",
            )
        )

        # Create video directories
        try:
            os.mkdir(os.path.join(session_path, "jaw_trace"))
            os.mkdir(os.path.join(session_path, "tongue_trace"))
            os.mkdir(os.path.join(session_path, "whisker_trace"))
        except:
            pass
        df_entry = pd.DataFrame(columns=trial_df.columns)
        hit_count = 0
        for trial in range(mat["TrialOnsets_All"][i].shape[0]):
            df_entry["trial_number"] = [trial]
            df_entry["trial_type"] = [
                (
                    mat["HitIndices"][i][trial] * 1 * "Hit"
                    + mat["CRIndices"][i][trial] * 1 * "CR"
                    + mat["FAIndices"][i][trial] * 1 * "FA"
                    + mat["MissIndices"][i][trial] * 1 * "Miss"
                    + (mat["CompletedTrialIndices"][i][trial] == 0) * 1 * "EarlyLick"
                )
            ]
            hit_count += (df_entry["trial_type"] == "Hit") * 1
            df_entry["trial_onset"] = [mat["TrialOnsets_All"][i][trial]]
            df_entry["reaction_time_piezo"] = [mat["ReactionTimes_All"][i][trial]]
            df_entry["reaction_time_jaw"] = [mat["LickOnsets"][i][trial]]
            df_entry["stim"] = [mat["StimIndices_All"][i][trial]]
            df_entry["trial_active"] = [mat["ActiveIndices"][i][trial]]
            jaw_path = os.path.join(session_path, "jaw_trace", "trial_{}".format(trial))
            tongue_path = os.path.join(
                session_path, "tongue_trace", "trial_{}".format(trial)
            )
            whisker_path = os.path.join(
                session_path, "whisker_trace", "trial_{}".format(trial)
            )
            jaw = mat["Jaw_pos"][i][trial, :]
            tongue = mat["Tongue_pos"][i][trial, :]
            whisker = mat["Whisker_pos"][i][trial, :]
            df_entry["video_onset"] = [-1]
            df_entry["video_offset"] = [3]
            if save_video:
                np.save(jaw_path, jaw)
                np.save(tongue_path, tongue)
                np.save(whisker_path, whisker)
            df_entry["jaw_trace"] = [jaw_path]
            df_entry["tongue_trace"] = [tongue_path]
            df_entry["whisker_angle"] = [whisker_path]
            trial_df = pd.concat([trial_df, df_entry], ignore_index=True)
        # how much of total water the mouse have drink
        trial_df = trial_df.fillna(0)
        trial_df.to_csv(os.path.join(session_path, "trial_info"))
        if save_clusters_info:
            save_clusters(mat, session_path, i)


def save_clusters(mat, session_path, i):
    """Save the cluster information from the matlab file to the folder system

    Args:
        mat (dict): the matlab file loaded with mat73
        session_path (str): where to save the data
        i (int): mounse number
    """
    cluster_df = pd.DataFrame(
        columns=(
            "neuron_index",
            "area",
            "excitatory",
            "depth",
            "cluster",
            "firing_rate",
            "with_video",
        )
    )
    df_entry = pd.DataFrame(columns=cluster_df.columns)
    for neuron in range(len(mat["area"][i])):
        if mat["include"][i][neuron] == 0:
            return
        # based on the AP waveform and then a simple threshold clustering
        exc = mat["width"][i][neuron] > 0.275
        df_entry["neuron_index"] = [neuron]
        df_entry["excitatory"] = [exc]
        df_entry["area"] = [mat["area"][i][neuron]]
        df_entry["depth"] = [mat["depth"][i][neuron]]
        df_entry["cluster"] = [mat["cluster"][i][neuron]]
        spikes = mat["spikets"][i][neuron]
        firing_rate = spikes.shape[0] / (spikes[-1] - spikes[0])
        np.save(
            os.path.join(session_path, "neuron_index_{}".format(neuron)),
            spikes,
        )
        df_entry["firing_rate"] = [firing_rate]
        df_entry["with_video"] = [1]
        cluster_df = pd.concat([cluster_df, df_entry], ignore_index=True)
    cluster_df.to_csv(os.path.join(session_path, "cluster_info"))


def unified_cluster_table(path="datasets/DataFromVahid_expert"):
    """Aggregate the information of all clusters to a single csv file. This helps in order to make the unique
    assignment in our network.

    Args:
        path (str, optional): Where to find and save the cluster_information. Defaults to "datasets/DataFromVahid_expert".
    """
    cluster_information = pd.DataFrame(
        columns=(
            "session",
            "area",
            "excitatory",
            "firing_rate",
            "cluster_index",
            "with_video",
        )
    )
    dirs = os.listdir(path)
    if "cluster_information" in dirs:
        dirs.remove("cluster_information")
    df_entry = {}
    for dir in dirs:
        df = pd.read_csv(os.path.join(path, dir, "cluster_info"))
        df_entry = pd.DataFrame(columns=cluster_information.columns)
        df_entry["session"] = [dir]
        df.neuron_index = df.index.values
        df.to_csv(os.path.join(path, dir, "cluster_info"))
        for index, row in df.iterrows():
            df_entry["area"] = [row.area]
            df_entry["excitatory"] = [row.excitatory]
            df_entry["firing_rate"] = [row.firing_rate]
            df_entry["cluster_index"] = [index]
            df_entry["with_video"] = [row.with_video]
            cluster_information = pd.concat(
                [cluster_information, df_entry], ignore_index=True
            )
    cluster_information.to_csv(os.path.join(path, "cluster_information"))


if __name__ == "__main__":
    save_datastructure()
    unified_cluster_table()
