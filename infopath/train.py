import torch
import numpy as np
import os
from optparse import OptionParser
from datasets.dataloader import balance_trial_type, keep_trial_types_behaviour
from infopath.model_loader import load_model_and_optimizer
from infopath.config import get_opt, save_opt, config_vahid, config_pseudodata
from utils.functions import load_data, return_trial_type, trial_metric, trial_type_perc
from utils.logger import get_log_path, init_results, log_model_and_results_gan
from utils.plot_utils import plot_rsnn_activity
from infopath.lick_classifier import prepare_classifier
from infopath.losses import discriminator_loss


def train(opt, model, netD, optimizerG, optimizerD, step=-1):
    if opt.verbose:
        print(model)
    results = init_results(opt.log_path)
    early_stop, accuracy, previous_test_loss, t_trial_pearson_model = (
        0,
        torch.zeros(1),
        1000,
        0,
    )
    to_save_lists = {
        "total_train_loss": [],
        "neuron_loss": [],
        "trial_loss": [],
        "fr_loss": [],
        "cross_corr_loss": [],
        "t_trial_pearson": [],
        "tm_mle_loss": [],
    }

    # load data and prepare lick_classifier
    (
        train_spikes,
        train_jaw,
        session_info_train,
        test_spikes,
        test_jaw,
        session_info_test,
    ) = load_data(model)

    # we set the loop so at every iteration we choose a single stimulus condition all the stimuli so we add a second variable for train spikes,
    # we do that so our algorithm is running a step with all trials, one with only stim trials, and one with only non-stim trials and then repeat
    stim_cond = [opt.trial_types]
    if 0 in opt.stim:
        stim_cond.append([0, 1])
    if 1 in opt.stim:
        stim_cond.append([2, 3])

    data_spikes = train_spikes.clone()
    data_jaw = None
    filt_jaw_test = None
    filt_jaw_train = None
    lick_classifier = None
    if opt.with_behaviour:
        filt_jaw_train = model.filter_fun1(train_jaw)
        filt_jaw_test = model.filter_fun1(test_jaw)
        data_jaw = train_jaw.clone()
        lick_classifier = prepare_classifier(
            filt_jaw_train,
            filt_jaw_test,
            session_info_train,
            session_info_test,
            opt.device,
        )
    t_trial_pearson_data = t_trial_pearson(
        model.filter_fun1(train_spikes),
        model.filter_fun1(test_spikes),
        model.filter_fun1(filt_jaw_train),
        model.filter_fun1(filt_jaw_test),
        model,
        session_info_train,
    )
    data_perc = trial_type_perc(
        session_info_train, num_trial_types=len(opt.trial_types)
    )
    if opt.verbose:
        print("T'_{trial} data train and test", t_trial_pearson_data)
        print("the data have trial type distribution: ", data_perc)

    while step < opt.n_steps:
        step += 1
        # prepare data for stimulus condition, loop with all, only stim, only no stim
        index = step % len(stim_cond)
        data_spikes, data_jaw, session_info = isolate_step_data(
            train_spikes,
            train_jaw,
            data_spikes,
            data_jaw,
            session_info_train,
            stim_cond[index],
            data_perc,
            opt.balance_trial_types,
        )
        torch.cuda.empty_cache()
        torch.manual_seed(step)
        all_stims = np.concatenate([stims for stims in session_info[1]])
        stims = all_stims[torch.randint(all_stims.shape[0], size=(opt.batch_size,))]
        stims = torch.tensor(stims, device=opt.device)
        model_spikes, voltages, model_jaw, state = model(stims)

        # Train Generator
        optimizerG.zero_grad()
        if opt.use_logits:
            model_activity = torch.sigmoid(voltages)
        else:
            model_activity = model_spikes
        (
            fr_loss,
            trial_loss,
            neuron_loss,
            cross_corr_loss,
            tm_mle_loss,
        ) = model.generator_loss(
            model_activity,
            data_spikes,
            model_jaw,
            data_jaw,
            session_info,
            netD,
        )
        total_train_loss = (
            neuron_loss + trial_loss + fr_loss + cross_corr_loss + tm_mle_loss
        )
        total_train_loss.backward()
        optimizerG.step()
        with torch.no_grad():
            model.rsnn.reform_recurent(opt.lr, l1_decay=opt.l1_decay)
            model.rsnn.reform_v_rest()

        # Train Discriminator
        if step % 5 == 0 and opt.gan_loss:
            optimizerD.zero_grad()
            if opt.t_trial_gan:
                model_spikes_gan = model.filter_fun2(model.filter_fun1(model_spikes))
                model_jaw_gan = model.filter_fun2(model.filter_fun1(model_jaw))
                data_spikes_gan = model.filter_fun2(model.filter_fun1(data_spikes))
                data_jaw_gan = model.filter_fun2(model.filter_fun1(data_jaw))
            else:
                model_spikes_gan = model_spikes
                model_jaw_gan = model_jaw
                data_spikes_gan = data_spikes
                data_jaw_gan = data_jaw
            lossD, accuracy = discriminator_loss(
                netD,
                model_spikes_gan,
                data_spikes_gan,
                data_jaw_gan,
                model_jaw_gan,
                session_info,
                model.rsnn.area_index,
                discriminator=True,
                t_trial_gan=opt.t_trial_gan,
                z_score=opt.z_score,
                trial_loss_area_specific=opt.trial_loss_area_specific,
            )
            lossD.backward()
            optimizerD.step()
        state = [[st.detach() for st in state[0]]]

        if opt.verbose:
            loss_str = f"total_loss {total_train_loss:.2f} "
            if opt.loss_neuron_wise:
                loss_str += f" | neuron_loss {neuron_loss:.2f}"
            if opt.loss_trial_wise:
                loss_str += f" | trial_loss {trial_loss:.2f}"
            if opt.loss_cross_corr:
                loss_str += f" | cross_corr {cross_corr_loss:.2f}"
            if opt.gan_loss:
                loss_str += f" | discriminator {accuracy.item():.2f}"
            if opt.loss_trial_matched_mle:
                loss_str += f" | tm_mle_loss {tm_mle_loss:.2f}"
            print("step", step, " ", loss_str)
            if step % (10 - (10 % len(stim_cond))) == 0:
                with torch.no_grad():
                    t_trial_pearson_model = t_trial_pearson(
                        model.filter_fun1(data_spikes),
                        model.filter_fun1(model_spikes),
                        model.filter_fun1(data_jaw),
                        model.filter_fun1(model_jaw),
                        model,
                        session_info,
                    )
                print("t_trial_pearson: ", t_trial_pearson_model)

        to_save_lists["total_train_loss"].append(float(total_train_loss))
        to_save_lists["neuron_loss"].append(float(neuron_loss))
        to_save_lists["trial_loss"].append(float(trial_loss))
        to_save_lists["fr_loss"].append(float(fr_loss))
        to_save_lists["cross_corr_loss"].append(float(cross_corr_loss))
        to_save_lists["tm_mle_loss"].append(float(tm_mle_loss))
        if t_trial_pearson_model != 0:
            to_save_lists["t_trial_pearson"].append(float(t_trial_pearson_model))

        if (
            step % (opt.log_every_n_steps - (opt.log_every_n_steps % len(stim_cond)))
            == 0
        ):
            torch.cuda.empty_cache()
            with torch.no_grad():
                torch.manual_seed(step)
                state = model.steady_state()
                input_spikes = model.input_spikes(stims)
                model.rsnn.sample_mem_noise(
                    input_spikes.shape[0], input_spikes.shape[1]
                )
                mem_noise = model.rsnn.mem_noise.clone()
                model_spikes, voltages, model_jaw, state = model.step(
                    input_spikes, state, mem_noise=mem_noise
                )
                activity_figure = prepare_plot(
                    model,
                    model_spikes,
                    voltages,
                    model_jaw,
                    input_spikes,
                    opt,
                    test_spikes,
                    test_jaw,
                    session_info_test,
                    lick_classifier,
                    stims,
                )
                (
                    test_loss,
                    trial_type_acc,
                    t_trial_pearson_ratio,
                ) = goodness_of_fit(
                    test_spikes,
                    model_spikes,
                    test_jaw,
                    model_jaw,
                    session_info_test,
                    stims,
                    model,
                    data_perc,
                    t_trial_pearson_data,
                    lick_classifier,
                    netD,
                )
                if opt.verbose:
                    print(
                        f"trial type accuracy {trial_type_acc} ratio of t_trial_test/t_trial_data {t_trial_pearson_ratio}"
                    )

            results["trial_type_accuracy"].append(trial_type_acc)
            results["t_trial_pearson_ratio"].append(t_trial_pearson_ratio.item())
            for key in to_save_lists.keys():
                results[key].append(np.mean(to_save_lists[key]))
            results["test_loss"].append(test_loss)

            log_model_and_results_gan(
                opt,
                model,
                optimizerG,
                netD,
                optimizerD,
                results,
                results["total_train_loss"][-1],
                step,
                [activity_figure],
            )
            for key in to_save_lists.keys():
                to_save_lists[key] = []
            if results["total_train_loss"][-1] > previous_test_loss:
                if early_stop > opt.early_stop:
                    print("probably pointless to continue so I stop myself")
                    print("So long, and thanks for the fish")
                    return -1
            else:
                previous_test_loss = results["total_train_loss"][-1]
                early_stop = 0
            torch.cuda.empty_cache()
        early_stop += 1


def t_trial_pearson(filt_data, filt_model, data_jaw, model_jaw, model, session_info):
    with torch.no_grad():
        t_trial_p = trial_metric(
            model.filter_fun2(filt_data),
            model.filter_fun2(filt_model),
            model.filter_fun2(data_jaw),
            model.filter_fun2(model_jaw),
            session_info,
            model,
            "pear_corr",
        )
        return t_trial_p.mean()


def prepare_plot(
    model,
    model_spikes,
    voltages,
    model_jaw,
    input_spikes,
    opt,
    data_spikes,
    data_jaw,
    session_info,
    lick_classifier=None,
    stims=None,
):
    filt_model = model.filter_fun1(model_spikes)
    filt_data = model.filter_fun1(data_spikes)
    filt_model_jaw = model.filter_fun1(model_jaw)
    filt_data_jaw = model.filter_fun1(data_jaw)
    # statistics of trial variability
    trial_type, model_perc = return_trial_type(
        model,
        filt_data,
        filt_model,
        filt_data_jaw,
        filt_model_jaw,
        session_info,
        stims,
        lick_classifier,
    )
    if opt.verbose:
        print("trial type distribution for the model", model_perc)

    # plot the mean neural activity per trial type and area
    mean_signals = []
    for tr_type in opt.trial_types:
        data_spikes_trty = test_hit_miss(data_spikes, session_info, trial_type=tr_type)
        exc, _ = model.mean_activity(data_spikes_trty)
        mean_signals.append(exc)

    for tr_type in opt.trial_types:
        if (trial_type == tr_type).sum() == 0:
            # plot zeros if the model predict nothing for that trial type
            signal = [np.zeros_like(sig) for sig in mean_signals[0]]
        else:
            signal, _ = model.mean_activity(model_spikes[:, trial_type == tr_type])
        mean_signals.append(signal)

    firing_rates_target = model.firing_rate
    # calculating the baseline firing rate
    firing_rates = model_spikes[: model.trial_onset].mean(0).mean(0) / model.timestep

    activity_figure, _ = plot_rsnn_activity(
        input=input_spikes[:, -1, :].cpu().numpy(),
        spikes=model_spikes[:, -1, :].cpu().numpy(),
        voltages=voltages[:, -1, :].cpu().numpy(),
        output=mean_signals,
        jaw=[model_jaw, model_jaw],
        firing_rates=firing_rates.cpu().detach().numpy(),
        firing_rates_target=firing_rates_target,
        n_neuron_max=500,
        areas=opt.num_areas,
        dt=(opt.stop - opt.start) / mean_signals[0][0].shape[0],
    )

    return activity_figure


def goodness_of_fit(
    data_spikes,
    model_spikes,
    data_jaw,
    model_jaw,
    session_info,
    stims,
    model,
    data_perc,
    t_trial_pearson_data,
    lick_classifier=None,
    netD=None,
):
    opt = model.opt
    (
        fr_loss,
        trial_loss,
        neuron_loss,
        cross_corr_loss,
        tm_mle_loss,
    ) = model.generator_loss(
        model_spikes, data_spikes, model_jaw, data_jaw, session_info, netD
    )
    filt_model = model.filter_fun1(model_spikes)
    filt_data = model.filter_fun1(data_spikes)
    filt_model_jaw = model.filter_fun1(model_jaw)
    filt_data_jaw = model.filter_fun1(data_jaw)
    test_loss = (
        neuron_loss + trial_loss + fr_loss + cross_corr_loss + tm_mle_loss
    ).item()
    trial_type, model_perc = return_trial_type(
        model,
        filt_data,
        filt_model,
        filt_data_jaw,
        filt_model_jaw,
        session_info,
        stims,
        lick_classifier,
    )

    data_perc_95 = 1.96 * (data_perc * (1 - data_perc) / opt.batch_size) ** 0.5
    acc_trial_type = (np.abs((model_perc.numpy() - data_perc)) < data_perc_95).sum()
    acc_trial_type /= len(opt.trial_types)
    t_trial_pearson_test = t_trial_pearson(
        filt_data, filt_model, filt_data_jaw, filt_model_jaw, model, session_info
    )
    t_trial_pearson_ratio = t_trial_pearson_test / t_trial_pearson_data

    return test_loss, acc_trial_type, t_trial_pearson_ratio


def test_hit_miss(data_spikes, session_info, trial_type=1):
    T, trials, neurons = data_spikes.shape
    max_trials = max([(i == trial_type).sum() for i in session_info[0]])
    test_data = torch.ones(T, max_trials, neurons) * torch.nan
    for sess in range(len(session_info[0])):
        trial_types = np.where(session_info[0][sess] == trial_type)[0]
        idx = session_info[-1][sess]
        sess_data = data_spikes[..., idx][:, trial_types].cpu()
        resid_trials = max_trials - trial_types.shape[0]
        sess_data = torch.cat(
            [sess_data, torch.ones(T, resid_trials, idx.sum()) * torch.nan], dim=1
        )
        test_data[:, :, session_info[-1][sess]] = sess_data
    return test_data


def isolate_step_data(
    train_spikes,
    train_jaw,
    data_spikes,
    data_jaw,
    session_info_train,
    stim,
    data_perc,
    balance_trial_types=False,
):
    data_spikes, data_jaw, session_info = keep_trial_types_behaviour(
        train_spikes,
        train_jaw,
        data_spikes,
        data_jaw,
        session_info_train,
        stim,
    )

    if balance_trial_types:
        data_spikes, data_jaw, session_info = balance_trial_type(
            data_spikes,
            data_jaw,
            session_info,
            data_perc[stim] / data_perc[stim].sum(),
        )
    return data_spikes, data_jaw, session_info


def git_diff(log_path):
    stream = os.popen("git diff infopath models")
    output = stream.read()
    with open(log_path + "/diff.txt", "w") as f:
        f.writelines(output)


if __name__ == "__main__":
    # here either you set the config from a file that you create with >> python3 infopath/config.py --config=$NAME
    # if there is no arg in the parser then is what is currently the default in the infopath/config.py
    parser = OptionParser()
    parser.add_option("--config", type="string", default="none")
    (pars, _) = parser.parse_args()

    if pars.config == "none":
        opt = config_pseudodata()
        # opt = config_vahid()
        get_log_path(opt, "trial")
    else:
        opt = get_opt(os.path.join("configs", pars.config))
        get_log_path(opt, pars.config)
    import warnings

    warnings.filterwarnings("ignore")
    if opt.verbose:
        print("log_path", opt.log_path)

    git_diff(opt.log_path)

    if pars.config != "none":
        os.mkdir(os.path.join(opt.log_path, pars.config))
        save_opt(os.path.join(opt.log_path, pars.config), opt)

    # set random seeds
    if opt.seed >= 0:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)

    # load model
    dev = opt.device
    opt.device = torch.device("cpu")
    model, netD, optimizerG, optimizerD = load_model_and_optimizer(opt)
    opt.device = dev
    model.to(opt.device)
    if netD is not None:
        netD.to(opt.device)
    train(opt, model, netD, optimizerG, optimizerD)
