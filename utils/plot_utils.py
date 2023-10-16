import json
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from matplotlib.patches import FancyArrowPatch
from umap import UMAP
from sklearn.decomposition import PCA
from utils.functions import *
from infopath.model_loader import load_model_and_optimizer

font = {"weight": "normal", "size": 6, "family": "sans-serif"}
import matplotlib

matplotlib.rc("font", **font)


def raster_plot(ax, spikes, dt=1):
    n_t, n_n = spikes.shape
    event_time_ids, event_ids = np.where(spikes)
    max_spike = 10000
    event_time_ids = event_time_ids[:max_spike] * dt
    event_ids = event_ids[:max_spike]
    ax.scatter(event_time_ids, event_ids, marker="|")
    ax.set_ylim([0 - 0.5, n_n - 0.5])
    ax.set_xlim([0, n_t])


def strip_right_top_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_activity(activity, ax, dt):
    colors = ["red", "black", "blue", "green"]
    trial_types = len(activity) // 2
    if type(activity) != list:
        activity = [activity]
    for i, act in enumerate(activity):
        time_line = np.arange(act.shape[0]) * dt
        ls = "-" if i < trial_types else "--"
        ax.plot(time_line, act, alpha=0.5, color=colors[i % trial_types], ls=ls)


def plot_rsnn_activity(
    input=None,
    spikes=None,
    voltages=None,
    output=None,
    jaw=None,
    dt=1.0,
    input_dt=1.0,
    n_neuron_max=50,
    n_voltages_max=2,
    firing_rates=None,
    firing_rates_target=None,
    areas=1,
):
    n_axis = 0
    if input is not None:
        n_axis += 1
    if output is not None:
        n_axis += areas
    if firing_rates is not None:
        n_axis += 1
    if spikes is not None:
        n_axis += 1
    if voltages is not None:
        n_axis += 1
    if jaw is not None:
        n_axis += 1

    ratio = 2 + 2 * n_axis
    fig, ax_list = plt.subplots(n_axis, 1, figsize=(6, ratio))
    current_ax_index = 0
    if input is not None:
        ax = ax_list[current_ax_index]
        raster_plot(ax, input)
        ax.set_ylabel("inputs")
        tmax = (input.shape[0] + 1) * input_dt
        ax.set_xlim([0, tmax])
        current_ax_index += 1

    if spikes is not None:
        n_neuron_show = n_neuron_max
        if n_neuron_max > spikes.shape[1]:
            n_neuron_show = spikes.shape[1]
        ax = ax_list[current_ax_index]
        s = spikes[:, : n_neuron_show // 2]
        s = np.concatenate([s, spikes[:, -n_neuron_show // 2 :]], axis=1)
        raster_plot(ax, s, dt=dt)
        tmax = (s.shape[0] + 1) * dt
        ax.set_xlim([0, tmax])
        current_ax_index += 1

    if voltages is not None:
        ax = ax_list[current_ax_index]
        v_choice = torch.randperm(voltages.shape[1])[:n_voltages_max]
        v = voltages[:, v_choice]
        s = spikes[:, v_choice]
        time_line = np.arange(v.shape[0]) * dt
        ax.plot(time_line, v)
        for i in range(min(v.shape[1], n_voltages_max)):
            v_i = v[:, i]
            s_i = s[:, i]
            time_indices = np.where(s_i > 0)[0]
            for t_idx_spikes in time_indices:
                v_spike_min = v_i[t_idx_spikes - 1]
                v_spike_max = v_spike_min + 2
                ax.vlines(
                    time_line[t_idx_spikes], v_spike_min, v_spike_max, color="black"
                )
        tmax = (s.shape[0] + 1) * dt
        ax.set_xlim([0, tmax])
        current_ax_index += 1

    if output is not None:
        num_signals = len(output)
        for area in range(areas):
            ax = ax_list[current_ax_index]
            try:
                pearcorr = pearsonr(output[0][area], output[num_signals // 2][area])[0]
            except:
                pearcorr = 0.0
            plot_activity([out[area] for out in output], ax, dt)
            ax.legend(["predicted", "True / pearson:{:1.3f}".format(pearcorr)])
            ax.set_ylabel("outputs area number{}".format(area))
            tmax = (output[1][area].shape[0] + 1) * dt
            ax.set_xlim([0, tmax])
            current_ax_index += 1

    if jaw is not None:
        ax = ax_list[current_ax_index]
        ax.plot(time_line, jaw[0][:, :5, 0].cpu(), "k")
        ax.plot(time_line, jaw[1][:, :5, 0].cpu(), "r")
        ax.set_xlim([0, tmax])
        current_ax_index += 1
    ax.set_xlabel("time in ms")

    for ax in ax_list:
        strip_right_top_axis(ax)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize("small")
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize("small")

    for ax in ax_list[:-2]:
        ax.set_xticks([])
    ax = ax_list[current_ax_index]
    current_ax_index += 1

    if firing_rates_target is not None:
        pearcorr = pearsonr(firing_rates_target, firing_rates)[0]
        ax.plot(firing_rates, alpha=0.5)
        ax.plot(firing_rates_target, alpha=0.5)
        ax.legend(["predicted", "True / pearson:{:1.3f}".format(pearcorr)])

    ax.set_xlabel("Neuron Id")
    ax.set_ylabel("Firing rates (Hz)")

    return fig, ax_list


def plot_with_size(width=50, height=20, inch=False):
    plotsize = [
        (width - 0.282) / (1 + 24.4 * (not inch)),
        (height - 0.282) / (1 + 24.4 * (not inch)),
    ]  # [W, H]
    height_scale = 1.5  # Scale to account for the axis title, labels and ticks.
    width_scale = 1.5  # Scale to account for the axis title, labels and ticks.
    figsize = [width_scale * plotsize[0] / 0.75, height_scale * plotsize[1] / 0.75]
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    wscale = (
        plotsize[0] / figsize[0]
    )  # = (width_scale*figsize[0]/0.75)/width_scale = figsize[0]/0.75 which is our target size for Inkscape.
    hscale = plotsize[1] / figsize[1]
    ax = fig.add_axes([0.3, 0.3, wscale, hscale])
    # ax.tick_params(left=False, bottom=False)
    return fig, ax


def plots_with_size(numx=1, numy=1, width=50, height=20, inch=False):
    plotsize = [
        (width - 0.282) / (1 + 24.4 * (not inch)),
        (height - 0.282) / (1 + 24.4 * (not inch)),
    ]  # [W, H]
    height_scale = 1.5  # Scale to account for the axis title, labels and ticks.
    width_scale = 1.5  # Scale to account for the axis title, labels and ticks.
    figsize = [width_scale * plotsize[0] / 0.75, height_scale * plotsize[1] / 0.75]
    print(figsize, plotsize)
    fig, ax = plt.subplots(
        numx,
        numy,
        figsize=(figsize[0] * numy, figsize[1]),
        gridspec_kw={"width_ratios": [1 for _ in range(numy)]},
    )
    # wscale = (
    #     plotsize[0] / figsize[0]
    # )  # = (width_scale*figsize[0]/0.75)/width_scale = figsize[0]/0.75 which is our target size for Inkscape.
    # hscale = plotsize[1] / figsize[1]
    for i in range(numx * numy):
        #     ax[i].figure.set_size_inches(
        #         plotsize[0] * (1 + 24.4 * (not inch)), plotsize[1] * (1 + 24.4 * (not inch))
        #     )
        ax[i].tick_params(left=False, bottom=False)
    return fig, ax


def loss_plot(log_paths, entry="t_trial_pearson"):
    fig, ax = plt.subplots()
    for i, log_path in enumerate(log_paths):
        results = json.load(open(log_path + "/results.json", "rb"))
        if "neuron_loss" not in results.keys():
            results["neuron_loss"] = results["neuron_loss"]
        if "total_train_loss" not in results.keys():
            results["total_train_loss"] = results["train_loss"]
        step = np.arange(len(results["test_loss"])) * 100
        # ax.plot(step, results["train_loss"], label="train" * (i == 0), color=f"C{i}")
        # ax.plot(step, results["test_loss"], label="test" * (i == 0), color=f"C{i}")
        # ax.plot(step, results["neuron_loss"], label="psth" * (i == 0), color=f"C{i}")
        # ax.plot(step, results["trial_loss"], label="pop_avg" * (i == 0), color=f"C{i}")
        # ax.plot(step, results["fr_loss"], label="fr_loss" * (i == 0), color=f"C{i}")
        ax.plot(step, results[entry], label=entry * (i == 0), color=f"C{i}")
        ax.legend()
        ax.set_yscale("log")
        print(f"C{i},  {log_path}")
    return fig, ax


def retrieve_spikes_trial_type(
    data_spikes, session_info, trial_type=1, active_quiet=None
):
    T, trials, neurons = data_spikes.shape
    max_trials = max([(i == trial_type).sum() for i in session_info[0]])
    test_data = torch.ones(T, max_trials, neurons) * torch.nan
    for sess in range(len(session_info[0])):
        condition = session_info[0][sess] == trial_type
        if active_quiet is not None:
            condition = condition & (session_info[2][sess] == active_quiet)
        trial_types = np.where(condition)[0]
        idx = session_info[-1][sess]
        sess_data = data_spikes[..., idx][:, trial_types].cpu()
        resid_trials = max_trials - trial_types.shape[0]
        sess_data = torch.cat(
            [sess_data, torch.ones(T, resid_trials, idx.sum()) * torch.nan], dim=1
        )
        test_data[:, :, session_info[-1][sess]] = sess_data
    return test_data


def retrieve_jaw_trial_type(jaw, session_info, trial_type=1, active_quiet=None):
    T, trials, sessions = jaw.shape
    max_trials = max([(i == trial_type).sum() for i in session_info[0]])
    test_data = torch.ones(T, max_trials, sessions) * torch.nan
    for sess in range(len(session_info[0])):
        condition = session_info[0][sess] == trial_type
        if active_quiet is not None:
            condition = condition & (session_info[2][sess] == active_quiet)
        trial_types = np.where(condition)[0]
        sess_data = jaw[..., sess][:, trial_types].cpu()
        resid_trials = max_trials - trial_types.shape[0]
        sess_data = torch.cat(
            [sess_data, torch.ones(T, resid_trials) * torch.nan], dim=1
        )
        test_data[:, :, sess] = sess_data
    return test_data


def retrieve_spikes_active_quiet(data_spikes, session_info, active=1):
    T, trials, neurons = data_spikes.shape
    max_trials = max([(i == active).sum() for i in session_info[2]])
    test_data = torch.ones(T, max_trials, neurons) * torch.nan
    for sess in range(len(session_info[2])):
        active_quiet = np.where(session_info[2][sess] == active)[0]
        idx = session_info[-1][sess]
        sess_data = data_spikes[..., idx][:, active_quiet].cpu()
        resid_trials = max_trials - active_quiet.shape[0]
        sess_data = torch.cat(
            [sess_data, torch.ones(T, resid_trials, idx.sum()) * torch.nan], dim=1
        )
        test_data[:, :, session_info[-1][sess]] = sess_data
    return test_data


def retrieve_hit_active(
    data_spikes, session_info, trial_type="both", active_quiet="both"
):
    filt_spikes = torch.ones_like(data_spikes) * torch.tensor(np.nan)
    for session in range(len(session_info[0])):
        idx = session_info[-1][session]
        to_keep = torch.where(~torch.isnan(data_spikes[..., idx].sum((0, 2))))[0]
        to_keep = to_keep.cpu().numpy()
        if trial_type != "both":
            to_keep = np.intersect1d(
                to_keep, np.where(session_info[0][session] == trial_type.item())[0]
            )
        if active_quiet != "both":
            to_keep = np.intersect1d(
                to_keep, np.where(session_info[2][session] == active_quiet.item())[0]
            )
        cond_spikes = data_spikes[:, to_keep][..., idx]
        filt_spikes[:, : to_keep.shape[0]][..., idx] = cond_spikes
    return filt_spikes


def make_spikes(
    model,
    batch_size=200,
    seed=0,
    session_info=None,
    stim=1,
    return_jaw=False,
    stims=None,
):
    with torch.no_grad():
        model.opt.batch_size = batch_size
        if stims is None:
            if session_info is None:
                stims = torch.ones(model.opt.batch_size).long().cuda() * stim
            else:
                stims = torch.zeros(batch_size)
                stims[: int(batch_size / 2)] = 1
                stims = stims[torch.randperm(batch_size)].long().to(model.opt.device)
        if seed is not None:
            torch.manual_seed(seed)
        state = model.steady_state()
        input_spikes = model.trialToSpike(stims)
        if seed is not None:
            torch.manual_seed(seed)
        model.rnn.cells[0].sample_mem_noise(
            input_spikes.shape[0], input_spikes.shape[1]
        )
        mem_noise = model.rnn.cells[0].mem_noise.clone()
        if seed is not None:
            torch.manual_seed(seed)
        spikes, v, jaw, _ = model.step_with_dt(
            input_spikes,
            state,
            sample_mem_noise=False,
            dt=25,
        )

        model.rnn.cells[0].mem_noise = mem_noise
        torch.cuda.empty_cache()
    if return_jaw:
        return stims, input_spikes, [spikes], jaw, [v]
    else:
        return stims, input_spikes, [spikes], [v]


def simulation_plots(model, opt):
    if opt.lsnn_version == "srm":
        w_rec = (model.rnn.cells[0].conv_rec.weight.cpu().detach() ** 2).mean(2)
    else:
        w_rec = model.rnn.cells[0]._w_rec.cpu().detach().clone()
    # w_rec[w_rec > 0.2] = 0.2
    if w_rec.dim() == 3:
        delays = model.rnn.cells[0].delays
        for i in range(w_rec.shape[0]):
            w_rec_tmp = w_rec[i]
            w_rec_fig, w_rec_ax = plt.subplots()
            delay = ((opt.inter_delay - delays[i] * opt.dt) // opt.dt * opt.dt)[i]
            #         delay = opt.inter_delay - i*opt.dt
            w_rec_ax.title(f"Delay {delay}")
            w_rec_ax.pcolormesh(w_rec_tmp, cmap=plt.get_cmap("gist_heat_r"))
            w_rec_ax.colorbar()

            areas = model.rnn.cells[0].areas
            exc = model.rnn.cells[0].excitatory
            inh = model.rnn.cells[0].inhibitory
            for i in range(areas + 1):
                w_rec_ax.axvline(int(i * exc / areas), color="black")
                w_rec_ax.axhline(int(i * exc / areas), color="black")
                w_rec_ax.axvline(exc + int(i * inh / areas), color="black")
                w_rec_ax.axhline(exc + int(i * inh / areas), color="black")

    else:
        w_rec_fig, w_rec_ax = plt.subplots()
        w_rec_ax.pcolormesh(w_rec, cmap=plt.get_cmap("gist_heat_r"))
        w_rec_ax.colorbar()
        areas = model.rnn.cells[0].areas
        exc = model.rnn.cells[0].excitatory
        inh = model.rnn.cells[0].inhibitory
        for i in range(areas + 1):
            w_rec_ax.axvline(int(i * exc / areas), color="black")
            w_rec_ax.axhline(int(i * exc / areas), color="black")
            w_rec_ax.axvline(exc + int(i * inh / areas), color="black")
            w_rec_ax.axhline(exc + int(i * inh / areas), color="black")

    w_inp_fig, w_inp_ax = plt.subplots()
    if opt.lsnn_version == "srm":
        w_inp = (model.rnn.cells[0].conv_inp.weight.cpu().detach()).mean(2)
        #     w_inp[w_inp<0] = 0
        bias = model.rnn.cells[0].conv_inp.bias.cpu().detach()
        #     w_inp[w_inp<0.5*1e-5] = 0.5*1e-5
        noise_bias = 1
    else:
        w_inp = model.rnn.cells[0]._w_in.cpu().detach()
        bias = model.rnn.cells[0].v_rest.cpu().detach()
        noise_bias = model.rnn.cells[0].bias.cpu().detach()

    w_inp_ax.pcolormesh(w_inp, cmap=plt.get_cmap("gist_heat_r"))
    w_inp_ax.colorbar()

    areas = model.rnn.cells[0].areas
    exc = model.rnn.cells[0].excitatory
    inh = model.rnn.cells[0].inhibitory
    for i in range(areas + 1):
        w_inp_ax.axhline(int(i * exc / areas), color="black")
        w_inp_ax.axhline(exc + int(i * inh / areas), color="black")
    w_inp_ax.axvline(opt.n_rnn_in // 2, color="black")

    bias_fig, bias_ax = plt.subplots()
    bias_ax.set_title("Bias")
    bias_ax.plot(bias)
    for i in range(areas + 1):
        bias_ax.axvline(int(i * exc / areas), color="black")
        bias_ax.axvline(exc + int(i * inh / areas), color="black")

    n_bias_fig, n_bias_ax = plt.subplots()
    n_bias_ax.set_title("Noise Bias")
    n_bias_ax.plot(noise_bias)
    for i in range(areas + 1):
        n_bias_ax.axvline(int(i * exc / areas), color="black")

        n_bias_ax.axvline(exc + int(i * inh / areas), color="black")

    offset_fig, offset_ax = plt.subplots()
    offset_ax.set_title("Trial Offset")
    trial_offset = model.rnn.cells[0].trial_offset.T.cpu().detach().clone()
    offset_ax.plot(trial_offset.abs())
    for i in range(areas + 1):
        offset_ax.axvline(int(i * exc / areas), color="black")
        offset_ax.axvline(exc + int(i * inh / areas), color="black")

    fr_fig, fr_ax = plt.subplots()
    fr_ax.set_title("Firing Rate")
    fr_ax.plot(model.firing_rate)
    for i in range(areas + 1):
        fr_ax.axvline(int(i * exc / areas), color="black")
        fr_ax.axvline(exc + int(i * inh / areas), color="black")

    sess_fig, sess_ax = plt.subplots()
    sess_ax.plot(model.sessions)
    for i in range(areas + 1):
        sess_ax.axvline(int(i * exc / areas), color="black")
        sess_ax.axvline(exc + int(i * inh / areas), color="black")
    return w_rec_fig, w_inp_fig, bias_fig, n_bias_fig, offset_fig, fr_fig, sess_fig


def light_generator(time, window="baseline", freq=100):
    time = torch.tensor(time)
    t = torch.arange(time.shape[0]) * 2 / 1000
    light = torch.sin(2 * t * torch.pi * freq) >= 0
    if window == "baseline":
        main = (-1 <= time) & (time <= -0.2)
        ramp = (-0.2 < time) & (time <= -0.1)
    elif window == "whisker":
        main = (-0.1 <= time) & (time <= 0.1)
        ramp = (0.1 < time) & (time <= 0.2)
    elif window == "delay":
        main = (0.2 <= time) & (time <= 0.9)
        ramp = (0.9 < time) & (time <= 1)
    elif window == "response":
        main = (1 <= time) & (time <= 2)
        ramp = (2 < time) & (time <= 2.1)
    else:
        assert False, "window name wrong"
    main = main * 1.0
    start_ramp = (ramp * 1.0).argmax()
    stop_ramp = start_ramp + ramp.sum()
    t = torch.arange(time.shape[0])
    if stop_ramp != start_ramp:
        ramp = ramp * (stop_ramp - t) / (stop_ramp - start_ramp)
        total = main + ramp
    else:
        total = main
    light = light * total
    return light, total


def light_area(light, batch_size, model, area):
    assert area < model.opt.num_areas, "wrong area"
    device = model.opt.device
    light_source = torch.zeros(
        model.opt.n_units, batch_size, light.shape[0], device=device
    )
    light = light.to(model.opt.device)
    light_source[model.rnn.cells[0].area_index == area, :, :] = light
    light_source = light_source.permute(2, 1, 0) * 20 * model.rnn.cells[0].base_thr[0]
    return light_source


def show_input_weights_per_area(
    weights, area_index, exc_index, opt, path="RSNN_Figures/Figure3", title="whisker"
):
    df = pd.DataFrame(columns=("weight", "exc", "area"))
    for i in range(weights.shape[0]):
        df_entry = {}
        df_entry["weight"] = weights[i].mean().item()
        df_entry["exc"] = "excitatory" if exc_index[i] else "inhibitory"
        df_entry["area"] = opt.areas[area_index[i]]
        df = df.append(df_entry, ignore_index=True)
    x = "area"
    y = "weight"
    order = opt.areas

    fig, ax = plot_with_size(120, 200)
    ax = sns.boxplot(ax=ax, x="area", y="weight", data=df, whis=1.5)
    ax.set_ylabel("Total incoming weights", fontsize=9)
    ax.set_xlabel("Area", fontsize=9)

    box_pairs = [(i, j) for k, j in enumerate(opt.areas) for i in opt.areas[:k]]
    add_stat_annotation(
        ax,
        data=df,
        x=x,
        y=y,
        order=order,
        box_pairs=box_pairs,
        test="Mann-Whitney",
        text_format="star",
        verbose=0,
        loc="inside",
    )
    ax.set_ylim(0, 0.1)

    fig.savefig(f"{path}/{title}_weights.pdf")


def input_current_rec(model, spikes, limits_cur, opt, dt=0):
    w_rec = model.rnn.cells[0]._w_rec.detach().clone()[dt]
    inp_curs = torch.zeros(spikes.shape[0], opt.num_areas * 2, opt.num_areas * 2)
    total_inp_cur = torch.einsum("ij,tbj->tbi", w_rec, spikes)
    for i in range(opt.num_areas * 2):
        for j in range(opt.num_areas * 2):
            inp_cur = torch.einsum(
                "ij, tbj->tbi",
                w_rec[
                    limits_cur[i] : limits_cur[i + 1], limits_cur[j] : limits_cur[j + 1]
                ],
                spikes[..., limits_cur[j] : limits_cur[j + 1]],
            )
            inp_cur = inp_cur.sum((1, 2))
            # / inp_cur[
            #     :, :, limits_cur[i] : limits_cur[i + 1]
            # ].sum((1, 2))
            inp_curs[:, i, j] = inp_cur
    return inp_curs


def input_current_df(onset, offset, inp_cur, opt, stim_time=0.2):
    onset = int(onset + stim_time / opt.dt / 0.001)
    offset = int((offset + stim_time) / opt.dt / 0.001)
    df = pd.DataFrame(columns=("start", "stop", "width"))
    df_entry = {}
    weights = inp_cur[onset:offset].mean(0)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            df_entry["start"] = (
                opt.areas[i % len(opt.areas)] + (i >= len(opt.areas)) * " Inh"
            )
            df_entry["stop"] = (
                opt.areas[j % len(opt.areas)] + (j >= len(opt.areas)) * " Inh"
            )
            df_entry["width"] = weights[i, j].item()
            df = df.append(df_entry, ignore_index=True)
    return df


def make_my_graph(
    ax,
    df,
    areas=["wS1", "wS2", "wM1", "wM2"],
    fontsize=6,
    radius=0.1,
    width_times=20,
    color="k",
):
    place = {area: i for i, area in enumerate(areas)}
    style = "Simple, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color=color, alpha=0.3)

    for row in df.itertuples():
        if (row.start not in areas) or (row.stop not in areas):
            continue
        start = place[row.start]
        stop = place[row.stop]
        if start != stop:
            off = radius / 2
            if stop > start:
                off = -radius / 2
            a = FancyArrowPatch(
                (0, start),
                (-off, stop + off),
                zorder=0,
                lw=row.width * width_times,
                connectionstyle="arc3,rad=0.3",
                **kw,
            )
        else:
            a = FancyArrowPatch(
                (0, start - radius),
                (0, stop + radius),
                zorder=0,
                lw=row.width * width_times,
                connectionstyle="arc3,rad=2.",
                **kw,
            )
        ax.add_patch0(a)

    for area in areas:
        start = place[area]
        circle = plt.Circle((0, start), radius, color="C0")
        ax.add_patch(circle)
        #         ax.scatter(0, start, s=300, color="C0")
        ax.text(-radius / 2, start, area, fontsize=fontsize)
    ylim_start = -0.5
    ylim_stop = len(place) + ylim_start
    ax.set_ylim(ylim_start, ylim_stop)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def split_session(session_info, area=None, area_index=None):
    if area is None:
        return session_info
    session_info_new = [[], [], [], []]
    for sess in range(len(session_info[0])):
        if (area_index[session_info[-1][sess]] == area).sum() == 0:
            continue
        for i in range(3):
            session_info_new[i].append(session_info[i][sess])
        idx = session_info[-1][sess] & (area_index == area)
        session_info_new[-1].append(idx)
    return session_info_new


def temp_active(data_spikes, session_info, area_index, trial_type=1, active_quiet=1):
    f_data = retrieve_spikes_trial_type(
        data_spikes, session_info, trial_type, active_quiet
    )
    T, K, N = f_data.shape
    temp_quiet = torch.zeros(T * 6)
    for i in range(6):
        temp_quiet[T * i : T * (i + 1)] = f_data[:, :, area_index == i].nanmean((1, 2))
    return temp_quiet


def pca_plot(
    model,
    spikes,
    data_spikes,
    session_info,
    data_jaw=None,
    jaw=None,
    pca=True,
    path=f"RSNN_Figures/Figure3/pca",
    us_model=None,
    trial_type=None,
    rand_state=0,
    zoomed=False,
):
    if zoomed:
        fig, ax = plot_with_size(34 * 4, 34 * 4)
    else:
        fig, ax = plot_with_size(34, 34)
    num_trial_types = len(model.opt.trial_types)
    area_index = model.rsnn.area_index
    num_areas = len(area_index.unique())

    data_template, mean, std = make_template(
        model,
        data_spikes,
        session_info,
        remove_mean=False,
        num_trial_types=num_trial_types,
        num_areas=num_areas,
        jaw=data_jaw,
    )

    colors = {0: "red", 1: "black", 2: "blue", 3: "green"}
    color = [colors[int(i)] for i in trial_type if not np.isnan(i)]

    flag_jaw = 0
    if jaw is not None:
        flag_jaw = 1
    data_template = data_template.reshape(
        num_areas + flag_jaw, spikes.shape[0], num_trial_types
    ).permute(1, 2, 0)
    spikes = model.filter_fun2(spikes)
    t_trial_data = model.filter_fun2(data_template).permute(2, 0, 1)
    T, K, N = spikes.shape
    if num_areas > 1:
        t_trial = torch.zeros((num_areas + flag_jaw) * T, K, device=model.opt.device)
        for area in range(num_areas):
            tmp = spikes[:, :, area_index == area].mean(2)
            t_trial[area * T : (area + 1) * T] = (tmp.T - mean[area]).T / std[area]
        t_trial_data = t_trial_data.reshape(-1, num_trial_types)
        if jaw is not None:
            area += 1
            jaw = model.filter_fun2(jaw)
            t_trial[area * T : (area + 1) * T] = (jaw[:, :, 0].T - mean[-1]).T / std[-1]
    else:
        t_trial = spikes.mean(2).detach()
        t_trial_data = t_trial_data[area]
    t_trial = torch.cat((t_trial, t_trial_data), dim=1).cpu().numpy()
    for i in range(num_trial_types):
        color.append(colors[i])

    if pca:
        if us_model is None:
            us_model = PCA().fit(t_trial.T)
        x = us_model.transform(t_trial.T)
        ax.scatter(x[:K, 0], x[:K, 1], color=color[:K], s=10 * 4, alpha=0.3)
        ax.scatter(
            x[K:, 0],
            x[K:, 1],
            s=60 * 4,
            marker="*",
            color=color[K:],
            edgecolor="white",
            linewidths=1,
        )
        ax.tick_params(axis="both", which="major", pad=-2)
    else:
        if us_model is None:
            us_model = UMAP(random_state=rand_state).fit(t_trial[:, :-4].T)
        x = us_model.transform(t_trial.T)

        ax.scatter(
            x[:K, 0], x[:K, 1], color=color[:K], s=60, alpha=0.3, edgecolor="white"
        )
        ax.scatter(
            x[K:, 0],
            x[K:, 1],
            s=500 / 3,
            marker="*",
            color=color[K:],
            edgecolor="white",
            linewidths=1,
        )
        ax.tick_params(axis="both", which="major", pad=-2)
    color = np.array(color)
    return us_model, x, color, fig, ax


def hit_miss_plot(
    ax,
    area,
    filt_model,
    filt_data,
    session_info,
    out,
    model,
    time_vector,
    colors=["black", "red"],
    mode="trial_type",
    classes=None,
    excitatory=None,
    remove_baseline=False,
):
    if classes is None:
        classes = out.unique()
    area_index = (model.rsnn.area_index == area).cpu()
    if excitatory is not None:
        if excitatory:
            area_index *= (model.rsnn.excitatory_index == 1).cpu()
        else:
            area_index *= (model.rsnn.excitatory_index == 0).cpu()
    if mode == "trial_type":
        titles = {0: "miss", 1: "hit", 2: "correct rejection", 3: "false alarm"}
    else:
        titles = {0: "quiet", 1: "active"}
    colors = {i: colors[i] for i in range(len(classes))}
    for i, o in enumerate(classes):
        if o == 250:
            continue
        if mode == "trial_type":
            f_data = retrieve_spikes_trial_type(filt_data, session_info, o)
        else:
            f_data = retrieve_spikes_active_quiet(filt_data, session_info, o)
        sig = f_data[:, :, area_index].nanmean((1, 2)).cpu()
        if remove_baseline:
            sig -= sig[:2].nanmean()
        ax.plot(
            time_vector,
            sig,
            label=f"data {titles[i]}",
            color=colors[i],
            alpha=0.3,
        )

        if out.dim() == 2:
            select = out[area_index].T
        else:
            select = out

        ax.axvline(0)

        sig = (
            filt_model[:, :, area_index][:, select == o]
            .cpu()
            .reshape(f_data.shape[0], -1)
            .mean(1)
        )
        if remove_baseline:
            sig -= sig[:2].nanmean()
        ax.plot(
            time_vector,
            sig,
            alpha=1,
            label=f"model {titles[i]}",
            color=colors[i],
        )
    return ax


def input_currents_over_time_plot(
    trial_types,
    spikes,
    model,
    opt,
    path="RSNN_Figures/Figure4/",
):
    exc = model.rnn.cells[0].excitatory
    inh = model.rnn.cells[0].inhibitory
    vlimits_exc = [int(i * exc / opt.num_areas) for i in range(opt.num_areas)]
    vlimits_inh = [int(i * inh / opt.num_areas) + exc for i in range(opt.num_areas)]
    vlimits_rec = vlimits_exc + vlimits_inh + [opt.n_units]
    condition = trial_types
    hit, miss = 0, 0
    for i in range(model.rnn.cells[0]._w_rec.shape[0]):
        hit += input_current_rec(
            model, spikes[0][:, condition == 1], vlimits_rec, opt, dt=i
        )
        miss += input_current_rec(
            model, spikes[0][:, condition == 0], vlimits_rec, opt, dt=i
        )
    times = [[0.0, 0.2], [0.2, 0.25], [0.3, 1.2], [1.2, 1.4]]
    period = ["baseline", "whisker", "delay", "response"]
    for i, t in enumerate(times):
        h = hit[int(t[0] / model.timestep) : int(t[1] / model.timestep), :6, :6].mean(0)
        # h = h / hit[:, :6, :6].mean((0, 2))
        norm = matplotlib.colors.CenteredNorm()
        fig, ax = plot_with_size(30, 26)
        im = ax.pcolormesh(h, cmap=plt.get_cmap("RdBu_r"), norm=norm)
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_xticklabels(model.opt.areas, rotation=45)
        ax.set_yticklabels(model.opt.areas, rotation=45)
        fig.savefig(
            f"{path}/inp_curr_hit_{period[i]}.svg", pad_inches=0, bbox_inches="tight"
        )
        fig, ax = plot_with_size(30, 26)
        m = miss[int(t[0] / model.timestep) : int(t[1] / model.timestep), :6, :6].mean(
            0
        )
        # m = m / miss[:, :6, :6].mean((0, 2))
        norm = matplotlib.colors.CenteredNorm()
        im = ax.pcolormesh(m, cmap=plt.get_cmap("RdBu_r"), norm=norm)
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_xticklabels(model.opt.areas, rotation=45)
        ax.set_yticklabels(model.opt.areas, rotation=45)
        fig.savefig(
            f"{path}/inp_curr_miss_{period[i]}.svg", pad_inches=0, bbox_inches="tight"
        )
        diff = h - m
        # diff = (diff / diff.sum(0).abs()).T
        fig, ax = plot_with_size(30, 26)
        norm = matplotlib.colors.CenteredNorm()
        im = ax.pcolormesh(diff, cmap="RdBu_r", norm=norm)
        # im = ax.pcolormesh(h - m, cmap=plt.get_cmap("RdBu_r"), norm=norm)
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_xticklabels(model.opt.areas, rotation=45)
        ax.set_yticklabels(model.opt.areas, rotation=45)

        fig.savefig(
            f"{path}/inp_curr_diff_{period[i]}.svg", pad_inches=0, bbox_inches="tight"
        )
    return ax


def metrics_plot(
    df,
    df_notm,
    measure="p",
    title="psth",
    path="RSNN_Figures/Figure3/",
    ylim=[0, 1],
    modulated=None,
):
    fig, ax = plot_with_size(27.6, 27.6)
    x = ["Test Model", "Shuffle Trials", "Shuffle Time"]
    p_mean_train = np.nanmean(df[df.pair == "Train Test"][measure].values[modulated])
    p_mean = [np.nanmean(df[df.pair == pair][measure].values[modulated]) for pair in x]
    p_std = [np.nanstd(df[df.pair == pair][measure].values[modulated]) for pair in x]
    x.append("No Trial Matching")
    p_mean.append(np.nanmean(df_notm[df_notm.pair == x[0]][measure].values[modulated]))
    p_std.append(np.nanstd(df_notm[df_notm.pair == x[0]][measure].values[modulated]))
    ax.axhline(p_mean_train, linestyle="--", color="black", label="Train Test")
    ax.errorbar(x, p_mean, p_std, color="blue", linestyle="", capsize=3)
    ax.scatter(x, p_mean, color="blue")
    ax.set_xticklabels(x, rotation=60)
    ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # if title == "pop":
    #     ax.spines["left"].set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     ax.legend()
    # else:
    #     ax.legend([])
    ax.tick_params(axis="both", which="major", pad=-2)
    fig.savefig(
        path + "statistics_" + title + ".svg", pad_inches=0, bbox_inches="tight"
    )


def miss_diff_table(df, title="", figurename="Figure4"):
    rows = ["whisker", "delay", "response"]
    columns = ["wS1", "wS2", "wM1", "wM2", "ALM", "tjM1"]
    misses = np.zeros((len(rows), len(columns)))
    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            misses[i, j] = df[(df.period == row) & (df.area == col)]["miss_diff"]
    fig, ax = plot_with_size(30, 26)
    # norm = matplotlib.colors.CenteredNorm()
    im = ax.pcolormesh(misses, cmap="RdBu_r", vmin=-0.41, vmax=0.41)
    fig.colorbar(im, ax=ax)
    ax.set_yticks([0.5 + (i) for i in range(len(rows))])
    ax.set_xticks([0.5 + (i) for i in range(len(columns))])
    ax.set_yticklabels(rows)
    ax.set_xticklabels(columns, rotation=45)
    fig.savefig(
        f"RSNN_Figures/{figurename}/{title}.png",
        pad_inches=0,
        bbox_inches="tight",
        dpi=400,
    )


def lick_prop_paper_like(df, area="wS1", remove_control=False):
    rows = df.period.unique()
    propabilities = np.zeros((2, len(rows)))
    for i, row in enumerate(rows):
        p = df[(df.period == row) & (df.area == area)]["propabilities"].values[0]
        propabilities[0, i] = p[1] / (p[0] + p[1] + 1e-6)  # hit rate
        propabilities[1, i] = p[3] / (p[2] + p[3] + 1e-6)  # fa rate
    # the number 0.66 and 0.206 is calculated from the data themselves
    prop_corr_hit = propabilities[0, :] - 0.66 * remove_control
    prop_corr_fa = propabilities[1, :] - 0.206 * remove_control
    return prop_corr_hit, prop_corr_fa


def lick_prop_with_ci(
    df,
    ax=None,
    area="wS1",
    hit_rate=True,
    remove_control=False,
):
    if hit_rate:
        color = "black"
    else:
        color = "red"

    with plt.rc_context({"font.size": 6 * 4}):
        windows = ["Control", "Stim", "Post", "Response"]
        hit_rate_area = np.zeros((4, 9))
        delta_hit_rate_area = np.zeros((4, 9))
        for j, window in enumerate(windows):
            control_cond = (df.area == area) & (df.win == "Control")
            cond = (df.area == area) & (df.win == window)
            df_subset = df.loc[cond | control_cond]
            if hit_rate:
                delta_hit = (
                    df_subset.loc[cond].Hit.values
                    - df_subset.loc[control_cond].Hit.values
                )
            else:
                delta_hit = (
                    df_subset.loc[cond].FA.values
                    - df_subset.loc[control_cond].FA.values
                )
            df_subset = df_subset.loc[cond]
            df_subset["delta_hit"] = delta_hit
            if hit_rate:
                hit_rate_area[j, :] = df_subset.groupby("mouse").Hit.mean().values
            else:
                hit_rate_area[j, :] = df_subset.groupby("mouse").FA.mean().values
            delta_hit_rate_area[j, :] = (
                df_subset.groupby("mouse").delta_hit.mean().values
            )

        hit_rate_area = np.array(hit_rate_area)
        if ax is not None:
            p = hit_rate_area.mean(1)
            # ci = 1.96 * np.sqrt((p * (1 - p)) / hit_rate_area.shape[1])
            ci = hit_rate_area.std(1) / (hit_rate_area.shape[1] ** 0.5)
            y = hit_rate_area.mean(1) - hit_rate_area.mean(1)[0] * remove_control
            # ax.scatter(np.arange(4), y, color=color)
            ax.errorbar(np.arange(4), y, yerr=ci, alpha=1, color=color, capsize=10)
            ax.scatter(np.arange(4), y, color=color)
            # ax.fill_between(np.arange(4), (y - ci), (y + ci), alpha=0.1, color=color)
        return hit_rate_area, delta_hit_rate_area


def pca_per_session(
    data_spikes,
    session_info,
    spikes,
    model,
    figurename,
    columns,
    column,
    figs_list=None,
    axs_list=None,
    extra="",
    pca=True,
):
    colors_data = {0: "red", 1: "black", 2: "blue", 3: "green"}
    colors = {0: "red", 1: "black", 2: "blue", 3: "green"}
    if figs_list is None:
        figs_list, axs_list = [], []
    for session in range(len(session_info[0])):
        filt_data, filt_model, idx = session_tensor(
            session, session_info, data_spikes, spikes
        )
        # min_trials = min(spikes_data.shape[1], spikes_model.shape[1])
        # spikes_data = spikes_data[:, :min_trials]
        # spikes_model = spikes_model[:, :min_trials]

        t_trial_data, t_trial_model = model.feature_pop_avg(
            filt_data, filt_model, 0, 0, idx, session
        )
        if t_trial_data == []:
            continue

        cost = mse_2d(t_trial_model.T, t_trial_data.T)
        idxx, idxy = linear_sum_assignment(cost.detach().cpu().numpy())
        t_trial_model = t_trial_model[: idxy.shape[0]]
        t_trial = torch.cat((t_trial_data, t_trial_model))
        if pca:
            us_model = PCA().fit(t_trial.cpu().numpy())
        else:
            us_model = UMAP(random_state=1).fit(t_trial.cpu().numpy())

        xpca_data = us_model.transform(t_trial_data.cpu().numpy())
        xpca_model = us_model.transform(t_trial_model.cpu().numpy())
        color = session_info[0][session]  # [:min_trials]
        colormodel = color[idxy]
        colormodel = [colors[i] for i in colormodel]
        color = [colors_data[i] for i in color]
        if len(figs_list) < session + 1:
            fig, ax = plt.subplots(1, columns, figsize=(5 * columns, 5))
        else:
            fig = figs_list[session]
            ax = axs_list[session]
        ax[0].set_title(
            f"Data, Session {session}, Num of Neurons: {filt_model.shape[2]}"
        )
        ax[0].scatter(
            xpca_data[:, 0],
            xpca_data[:, 1],
            s=50,
            # facecolor="None",
            color=color,
            alpha=0.3,
        )

        ymin, ymax = xpca_data[:, 1].min(), xpca_data[:, 1].max()
        xmin, xmax = xpca_data[:, 0].min(), xpca_data[:, 0].max()
        margin = np.abs(ymax)
        print(margin)
        ax[0].set_ylim(ymin - margin, ymax + margin)
        ax[0].set_xlim(xmin - margin, xmax + margin)
        # plt.savefig(f"RSNN_Figures/{figurename}/pca_session/{session}_data")

        ax[column].set_title(f"Model {extra}")
        ax[column].scatter(
            xpca_model[:, 0],
            xpca_model[:, 1],
            s=50,
            # facecolor="None",
            color=colormodel,
            alpha=0.3,
        )
        ax[column].set_ylim(ymin - margin, ymax + margin)
        ax[column].set_xlim(xmin - margin, xmax + margin)
        if pca:
            string = "pca"
        else:
            string = "umap"
        fig.savefig(f"RSNN_Figures/{figurename}/{string}_session/{session}_model")
        if len(figs_list) < session + 1:
            figs_list.append(fig)
            axs_list.append(ax)
    return figs_list, axs_list
