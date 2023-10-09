import torch
import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
from infopath.config import load_training_opt, save_opt


def get_log_path(opt, model_path=""):
    if hasattr(opt, "log_path"):
        return opt.log_path

    root_path = "log_dir"
    curr_commit = get_commit_path()
    root_path = os.path.join(root_path, curr_commit)
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    now = datetime.datetime.now()
    log_time = now.strftime("%Y_%-m_%-d_%-H_%-M_%-S")
    log_time += "_" + opt.task + "_" + model_path
    full_path = os.path.join(root_path, log_time)
    os.mkdir(full_path)
    opt.log_path = full_path
    save_opt(opt.log_path, opt)
    return opt.log_path


def get_commit_path():
    # you need to run the commmand from rssn2 directory
    with open(".git/HEAD", "r") as f:
        line = f.readline()
    branch = line.split("/")[-1][:-1]

    if os.path.exists(".git/logs/refs/remotes/origin/{}".format(branch)):
        with open(".git/logs/refs/remotes/origin/{}".format(branch), "r") as f:
            lines = f.readlines()
        curr_commit = lines[-1].split(" ")[1]
    else:
        curr_commit = branch
    return curr_commit


def init_results(log_path=None):
    try:
        results = json.load(open(os.path.join(log_path, "results.json"), "rb"))
    except:
        results = {
            "valid_accuracy": [],
            "total_train_loss": [],
            "train_loss": [],
            "train_acc": [],
            "step_at_epoch": [],
            "dt_forward": [],
            "dt_backward": [],
            "dt_step": [],
            "best_acc": -1,
            "best_acc_epoch": None,
            "best_acc_step": None,
            "test_loss": [],
            "best_loss": 100000000000000,
            "neuron_loss": [],
            "trial_loss": [],
            "tm_mle_loss": [],
            "fr_loss": [],
            "cross_corr_loss": [],
            "t_trial_pearson": [],
            "trial_type_accuracy": [],
            "opto_hit_rate": [],
            "t_trial_pearson_ratio": [],
        }

    return results


def save_results(log_path, results):
    result_path = os.path.join(log_path, "results.json")

    with open(result_path, "w") as f:
        json.dump(results, f)


def log_model_and_results(
    opt,
    model,
    optimizer,
    results,
    current_loss,
    step,
    activity_figure=None,
):
    log_path = opt.log_path
    model_path = os.path.join(log_path, "last_model.ckpt")
    optim_path = os.path.join(log_path, "last_optim.ckpt")
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optim_path)
    save_opt(log_path, opt)

    if activity_figure is not None:
        if isinstance(activity_figure, list):
            for i, activity in enumerate(activity_figure[:2]):
                act_type = "session" if i == 1 else "all_sessions"
                figure_path = os.path.join(log_path, f"activity_{act_type}_{step}.svg")
                activity.savefig(figure_path)
            if len(activity_figure) > 2:
                figure_path = os.path.join(log_path, f"loss_breakdown_{step}.svg")
                activity_figure[2].savefig(figure_path)
        else:
            activity_figure.savefig(os.path.join(log_path, f"activity_{step}.svg"))
    if "./datasets" != opt.datapath:
        np.save(os.path.join(log_path, "sessions"), model.sessions)
        np.save(os.path.join(log_path, "areas"), model.areas)
        np.save(os.path.join(log_path, "firing_rate"), model.firing_rate)
        np.save(os.path.join(log_path, "neuron_index"), model.neuron_index)

    if current_loss < results["best_loss"]:
        results["best_loss"] = current_loss
        model_path = os.path.join(log_path, "best_model.ckpt")
        torch.save(model.state_dict(), model_path)

    save_results(log_path, results)

    plot_save_weight_matrix(model, opt, step)
    plot_save_loss(results, opt.log_path)


def log_model_and_results_gan(
    opt,
    model,
    optimizer,
    netD,
    optimizerD,
    results,
    current_loss,
    step,
    activity_figure=None,
    save_all_matrices=False,
):
    log_path = opt.log_path
    model_path = os.path.join(log_path, "last_model.ckpt")
    optim_path = os.path.join(log_path, "last_optim.ckpt")
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optim_path)
    if netD is not None:
        netD_path = os.path.join(log_path, "last_netD.ckpt")
        optimD_path = os.path.join(log_path, "last_optimD.ckpt")
        torch.save(netD.state_dict(), netD_path)
        torch.save(optimizerD.state_dict(), optimD_path)
    save_opt(log_path, opt)

    if activity_figure is not None:
        if isinstance(activity_figure, list):
            for i, activity in enumerate(activity_figure[:2]):
                act_type = "session" if i == 1 else "all_sessions"
                figure_path = os.path.join(log_path, f"activity_{act_type}_{step}.svg")
                activity.savefig(figure_path)
            if len(activity_figure) > 2:
                figure_path = os.path.join(log_path, f"loss_breakdown_{step}.svg")
                activity_figure[2].savefig(figure_path)
        else:
            activity_figure.savefig(os.path.join(log_path, f"activity_{step}.svg"))
    if "./datasets" != opt.datapath:
        np.save(os.path.join(log_path, "sessions"), model.sessions)
        np.save(os.path.join(log_path, "areas"), model.areas)
        np.save(os.path.join(log_path, "firing_rate"), model.firing_rate)
        np.save(os.path.join(log_path, "neuron_index"), model.neuron_index)

    if current_loss < results["best_loss"]:
        results["best_loss"] = current_loss
        model_path = os.path.join(log_path, "best_model.ckpt")
        torch.save(model.state_dict(), model_path)

    best_tr_ty_acc, best_t_trial, best_opto = "", "", ""
    if len(results["trial_type_accuracy"]) > 1:
        if results["trial_type_accuracy"][-1] >= max(
            results["trial_type_accuracy"][:-1]
        ):
            best_tr_ty_acc = "_trial_type"
        if results["t_trial_pearson_ratio"][-1] >= max(
            results["t_trial_pearson_ratio"][:-1]
        ):
            best_t_trial = "_t_trial_ratio"
        if results["opto_hit_rate"] != []:
            if (
                results["opto_hit_rate"][-1] >= max(results["opto_hit_rate"][:-1])
                and results["opto_hit_rate"][-1] > 0
            ):
                best_opto = "_opto_hit"
    if best_tr_ty_acc + best_t_trial + best_opto != "":
        model_str = f"best{best_tr_ty_acc}{best_t_trial}{best_opto}_model.ckpt"
        model_path = os.path.join(log_path, model_str)
        with open(os.path.join(log_path, "model_info.txt"), "a") as f:
            f.writelines(f"{step}      {model_str} \n")
        torch.save(model.state_dict(), model_path)
    save_results(log_path, results)

    plot_save_weight_matrix(model, opt, step, save_all_matrices)
    plot_save_loss(results, opt.log_path)


def plot_save_weight_matrix(model, opt, step, save_all_matrices):
    extra = ""
    if save_all_matrices:
        extra = f"_{step}"
    log_path = opt.log_path
    w_rec = model.rsnn._w_rec.detach().cpu().numpy()
    w_in = model.rsnn._w_in.detach().cpu().numpy()
    if model.rsnn.motor_areas.shape[0] > 0:
        w_jaw = model.rsnn._w_jaw_post.detach().cpu().numpy().T

    exc = model.rsnn.excitatory
    inh = model.rsnn.inhibitory
    num_areas = opt.num_areas
    groups = opt.no_inter_intra * opt.rec_groups + (opt.no_inter_intra == 0)
    if opt.lsnn_version == "srm":
        groups = 1
    fig, ax = plt.subplots(groups, 1)
    if opt.no_inter_intra and groups == 1 and opt.lsnn_version != "srm":
        w_rec = w_rec[0]
    if opt.no_inter_intra and groups > 1:
        for g in range(groups):
            im = ax[g].pcolormesh(w_rec[g], cmap=plt.get_cmap("gist_heat_r"))
            for i in range(num_areas):
                ax[g].axhline((i + 1) * exc / num_areas, color="black")
                ax[g].axvline((i + 1) * exc / num_areas, color="black")
                ax[g].axhline(exc + (i + 1) * inh / num_areas, color="black")
                ax[g].axvline(exc + (i + 1) * inh / num_areas, color="black")
            fig.colorbar(im, ax=ax[g])
    else:
        for i in range(num_areas):
            ax.axhline((i + 1) * exc / num_areas, color="black")
            ax.axvline((i + 1) * exc / num_areas, color="black")
            ax.axhline(exc + (i + 1) * inh / num_areas, color="black")
            ax.axvline(exc + (i + 1) * inh / num_areas, color="black")
        im = ax.pcolormesh(w_rec, cmap=plt.get_cmap("gist_heat_r"))
        fig.colorbar(im)
    fig.savefig(os.path.join(log_path, f"rec_weight{extra}.png"))
    fig, ax = plt.subplots(1, 1)
    for i in range(num_areas):
        ax.axhline((i + 1) * exc / num_areas, color="black")
        ax.axhline(exc + (i + 1) * inh / num_areas, color="black")
    im = ax.pcolormesh(w_in, cmap=plt.get_cmap("gist_heat_r"))
    fig.colorbar(im)
    fig.savefig(os.path.join(log_path, f"input_weight{extra}.png"))

    if model.rsnn.motor_areas.shape[0] > 0:
        fig, ax = plt.subplots(1, 1)
        for i in range(num_areas):
            ax.axhline((i + 1) * exc / num_areas, color="black")
            ax.axhline(exc + (i + 1) * inh / num_areas, color="black")
        im = ax.pcolormesh(w_jaw, cmap=plt.get_cmap("gist_heat_r"))
        fig.colorbar(im)
        fig.savefig(os.path.join(log_path, "jaw_weight{extra}.png"))


def plot_save_loss(results, log_path):
    fig, ax = plt.subplots()
    ax.plot(results["total_train_loss"], label="total_train_loss")
    ax.plot(results["test_loss"], label="test_loss")
    ax.plot(results["fr_loss"], label="fr_loss")
    ax.plot(results["neuron_loss"], label="neuron_loss")
    ax.plot(results["trial_loss"], label="trial_loss")
    if "diff" in results.keys():
        ax.plot(results["diff"], label="mse params")
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(os.path.join(log_path, "losses.png"))
    del fig, ax


def reload_weights(training_opt, model, optimizer=None, last_best="last"):
    model_path = os.path.join(training_opt.log_path, "{}_model.ckpt".format(last_best))
    model.load_state_dict(
        torch.load(model_path, map_location=training_opt.device), strict=True
    )

    if optimizer is None:
        return model

    optim_path = os.path.join(training_opt.log_path, f"{last_best}_optim.ckpt")
    if not os.path.exists(optim_path):
        raise ValueError("optimizer not found at " + optim_path)
    optimizer.load_state_dict(torch.load(optim_path, map_location=training_opt.device))
    return model, optimizer


def go_back_to_hunting(log_path, last_best="last", lr=None):
    opt = load_training_opt(log_path)
    from infopath.model_loader import load_model_and_optimizer
    from infopath.train import train

    if last_best == "last":
        reload_model_and_optim = True
        reload_model = False
    else:
        reload_model_and_optim = False
        reload_model = True
    model, netD, optimizerG, optimizerD = load_model_and_optimizer(
        opt,
        reload_model_and_optim=reload_model_and_optim,
        reload_model=reload_model,
        last_best=last_best,
    )

    if lr is not None:
        optimizerG.param_groups[0]["lr"] = lr
        optimizerD.param_groups[0]["lr"] = lr
    steps = os.listdir(log_path)
    steps = [int(i.split("_")[-1][:-4]) for i in steps if ".svg" in i]
    steps = max(steps)
    optimizer_to(optimizerD, opt.device)
    optimizer_to(optimizerG, opt.device)
    train(opt, model, netD, optimizerG, optimizerD, step=steps)


# code from https://github.com/pytorch/pytorch/issues/8741
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
