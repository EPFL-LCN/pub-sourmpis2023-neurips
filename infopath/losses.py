import torch
from utils.functions import mse_2d, session_tensor, mse_2dv2, feature_pop_avg
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def firing_rate_loss(firing_rate, model_spikes, trial_onset, timestep):
    """Calculate the firing rate loss, the firing rate is calculated from the baseline activity."""
    data_fr = torch.tensor(firing_rate, device=model_spikes.device)
    model_fr = model_spikes[:trial_onset].mean(0).mean(0) / timestep
    return ((model_fr - data_fr) ** 2).mean()


def hard_trial_matching_loss(filt_data_spikes, filt_model_spikes):
    """Calculate the trial-matching loss with the hard matching (Hungarian Algorithm)
    Args:
        filt_data_spikes (torch.tensor): $\mathcal{T}_{trial}(z^\mathcal{D})$, dim Trials x Time
        filt_model_spikes (torch.tensor): $\mathcal{T}_{trial}(z)$, dim Trials x Time
    """
    # downsample the biggest tensor, so both data and model have the same #trials
    min_trials = min(filt_model_spikes.shape[0], filt_data_spikes.shape[0])
    filt_data_spikes = filt_data_spikes[:min_trials]
    filt_model_spikes = filt_model_spikes[:min_trials]
    with torch.no_grad():
        cost = mse_2d(filt_model_spikes.T, filt_data_spikes.T)
        keepx, ytox = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.nn.MSELoss()(filt_model_spikes[keepx], filt_data_spikes[ytox])


def trial_matched_mle_loss(
    data_spikes, model_spikes, session_info, data_jaw, model_jaw
):
    """Calculate the trial matched MLE, see Appendix figure 2"""
    loss = 0
    for session in range(len(session_info[0])):
        (data_spikes_sess, model_spikes_sess, _, _, _) = session_tensor(
            session, session_info, data_spikes, model_spikes, data_jaw, model_jaw
        )
        data_spikes_sess = (data_spikes_sess > 0) * 1.0
        mean = data_spikes_sess.mean((0, 1))
        std = data_spikes_sess.std((0, 1))
        data_spikes_sess = (data_spikes_sess - mean) / std
        model_spikes_sess = (model_spikes_sess - mean) / std
        # downsample the biggest tensor, so both data and model have the same #trials
        min_trials = min(model_spikes_sess.shape[1], data_spikes_sess.shape[1])
        T, K, N = model_spikes_sess.shape
        model_spikes_sess = model_spikes_sess[
            :, torch.randperm(K)[:min_trials]
        ].permute(1, 0, 2)
        T, K, N = data_spikes_sess.shape
        data_spikes_sess = data_spikes_sess[:, torch.randperm(K)[:min_trials]].permute(
            1, 0, 2
        )
        with torch.no_grad():
            cost = mse_2dv2(
                model_spikes_sess.permute(1, 0, 2),
                data_spikes_sess.permute(1, 0, 2),
            )
            keepx, ytox = linear_sum_assignment(cost.detach().cpu().numpy())
        loss += nn.MSELoss()(model_spikes_sess[:, keepx], data_spikes_sess[:, ytox])
    return loss


def cross_corr_loss(
    self,
    spikes_data_all,
    spikes_model_all,
    session_info,
    window=0.012,
    normalize=True,
):
    """Calculate the cross-correlation loss. Not used in this paper."""
    cross_corr = 0
    for session in range(len(session_info[0])):
        spikes_data, spikes_model, _, _, _ = session_tensor(
            session, session_info, spikes_data_all, spikes_model_all, None, None
        )
        min_trials = min(spikes_data.shape[1], spikes_model.shape[1])
        cross_corr_loss_sess = 0
        points = int(window / self.timestep)
        for delay in range(points):
            if delay < self.rsnn.n_delay:
                continue
            # in one pass we calculate both post-pre (upper diagonal) and pre-post (lower diagonal)
            spikes_now = spikes_data[:-delay][:, :min_trials].permute(1, 0, 2)
            spikes_dt = spikes_data[delay:][:, :min_trials].permute(1, 0, 2)
            if normalize:
                spikes_now = spikes_now - spikes_now.mean(0)
                spikes_dt = spikes_dt - spikes_dt.mean(0)
            spikes_now = spikes_now.reshape(-1, spikes_data.shape[2])
            spikes_dt = spikes_dt.reshape(-1, spikes_data.shape[2])
            cross_corr_data = spikes_dt.T @ spikes_now
            cross_corr_data.fill_diagonal_(0)

            spikes_now = spikes_model[:-delay][:, :min_trials].permute(1, 0, 2)
            spikes_dt = spikes_model[delay:][:, :min_trials].permute(1, 0, 2)
            if normalize:
                spikes_now = spikes_now - spikes_now.mean(0)
                spikes_dt = spikes_dt - spikes_dt.mean(0)
            spikes_dt = spikes_dt.reshape(-1, spikes_data.shape[2])
            spikes_now = spikes_now.reshape(-1, spikes_data.shape[2])
            cross_corr_model = spikes_dt.T @ spikes_now
            cross_corr_model.fill_diagonal_(0)
            cross_corr_loss_sess += ((cross_corr_data - cross_corr_model) ** 2).mean()
        cross_corr_loss_sess /= points
        cross_corr_loss_sess *= spikes_data.shape[2] / spikes_model_all.shape[2]
        cross_corr += cross_corr_loss_sess
    return cross_corr


def neuron_loss(filt_data, filt_model):
    psth_model, psth_data = z_score_norm(filt_data, filt_model)
    loss = ((psth_model - psth_data) ** 2).mean(0).mean()
    return loss


def z_score_norm(data_signal, model_signal, dim=1):
    # initial dim (x,y,z) sig = (sig-min)/(max-min), final dim (x,(!dim))
    data_meandim = data_signal.nanmean(dim)
    model_meandim = model_signal.nanmean(dim)
    mean = data_meandim.mean(0)
    std = data_meandim.std(0)
    std[std < 0.001] = 1
    model_meandim = (model_meandim - mean) / std
    data_meandim = (data_meandim - mean) / std
    return model_meandim, data_meandim


def discriminator_loss(
    netD,
    model_spikes,
    data_spikes,
    data_jaw,
    model_jaw,
    session_info,
    area_index,
    discriminator=True,
    t_trial_gan=True,
    z_score=True,
    trial_loss_area_specific=True,
):
    loss, sessions, accuracy = 0, 0, 0
    bce = torch.nn.BCELoss()
    output_data, output_model, labels_data, labels_model = [], [], [], []
    for session in range(len(session_info[0])):
        (
            filt_data_s,
            filt_model_s,
            f_data_jaw,
            f_model_jaw,
            idx,
        ) = session_tensor(
            session,
            session_info,
            data_spikes,
            model_spikes,
            data_jaw,
            model_jaw,
        )
        if t_trial_gan:
            feat_data, feat_model = feature_pop_avg(
                filt_data_s,
                filt_model_s,
                f_data_jaw,
                f_model_jaw,
                area_index[idx],
                session,
                z_score=z_score,
            )
        else:
            feat_data = filt_data_s
            feat_model = filt_model_s
            if f_data_jaw is not None:
                feat_data = torch.cat([feat_data, f_data_jaw])
                feat_model = torch.cat([feat_model, f_model_jaw])
        if len(feat_data) == 0:
            continue
        if discriminator:
            feat_model = feat_model.detach()

        if t_trial_gan:
            out_data = netD.discriminators[session](feat_data)
            out_model = netD.discriminators[session](feat_model)
        else:
            out_data = netD.discriminators[session](feat_data.permute(1, 2, 0))
            out_model = netD.discriminators[session](feat_model.permute(1, 2, 0))

        l_data = torch.ones_like(out_data)
        if discriminator:
            l_model = torch.zeros_like(out_model)
        else:
            l_model = torch.ones_like(out_model)
        output_data.append(out_data)
        output_model.append(out_model)
        labels_data.append(l_data)
        labels_model.append(l_model)
    output_data = torch.cat(output_data)
    output_model = torch.cat(output_model)
    labels_model = torch.cat(labels_model)
    labels_data = torch.cat(labels_data)

    if discriminator:
        loss = (bce(output_data, labels_data) + bce(output_model, labels_model)) / 2
    else:
        loss = bce(output_model, labels_model)
    labels = torch.cat((labels_model, labels_data))
    out = torch.cat((output_model, output_data))
    accuracy = (((out > 0.5) == (labels > 0)) * 1.0).mean()
    if discriminator:
        accuracy = roc_auc_score(
            labels.cpu().numpy() > 0,
            out.detach().cpu().numpy() > 0.5,
        )
    return loss, accuracy


def trial_matching_loss(
    filt_data_all,
    filt_model_all,
    session_info,
    data_jaw,
    model_jaw,
    loss_fun,
    area_index,
    z_score=True,
    trial_loss_area_specific=True,
):
    """Here we actually calculate the trial-matching loss function

    Args:
        filt_data_all (torch.tensor): filtered data spikes
        filt_model_all (torch.tensor): filtered model spikes
        session_info (list): information about different sessions
        data_jaw (torch.tensor): filtered data jaw trace
        model_jaw (torch.tensor): filtered model jaw trace
        loss_fun (function): function either hard_trial_matching_loss or sinkhorn loss
        area_index (torch.tensor): tensor with areas
        z_score (bool, optional): whether to z_score or not. Defaults to True.
        trial_loss_area_specific (bool, optional): Whether the T_trial is area specific. Defaults to True.

    Returns:
        _type_: _description_
    """
    loss, sessions = 0, 0
    # here we choose if we use soft or hard trial matching
    for session in range(len(session_info[0])):
        filt_data, filt_model, f_data_jaw, f_model_jaw, idx = session_tensor(
            session,
            session_info,
            filt_data_all,
            filt_model_all,
            data_jaw,
            model_jaw,
        )
        # if the session has less than 10 neurons or no trials(this happens only for
        #   particular trial type & stimulus conditions) don't take into account
        if filt_data.shape[2] < 10 or filt_model.shape[1] == 0:
            continue
        feat_data, feat_model = feature_pop_avg(
            filt_data,
            filt_model,
            f_data_jaw,
            f_model_jaw,
            area_index[idx],
            session,
            z_score=z_score,
            trial_loss_area_specific=trial_loss_area_specific,
        )
        if len(feat_data) == 0:
            continue
        # apply the proper scaling depending the method
        scaling = 1
        if loss_fun.__dict__ != {}:
            if loss_fun.loss != "energy":
                scaling = feat_data.shape[1]
            if loss_fun.p == 1:
                scaling = scaling**0.5
            scaling = feat_data.shape[1] ** 0.5
        loss += loss_fun(feat_data, feat_model) / scaling
        sessions += 1
    loss /= sessions
    return loss
