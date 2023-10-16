import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from datasets.dataloader import TrialDataset


def feature_pop_avg(
    filt_data,
    filt_model,
    filt_data_jaw,
    filt_model_jaw,
    area,
    session=0,
    z_score=False,
    trial_loss_area_specific=True,
):
    """Calculate the $\mathcal{T}'_\mathrm{trial}$ signals

    Args:
        filt_data (torch.tensor): Data with dim time x trials x neurons
        filt_model (torch.tensor): Model data with dim time x trials x neurons
        filt_data_jaw (torch.tensor): Data behaviour signalss with dim time x trials x session_id
        filt_model_jaw (torch.tensor): Model behaviour signals with dim time x trials x 1
        idx (np.array): the neurons from the current session
        session (int, optional): current session. Defaults to 0.
        z_score (bool, optional): Whether to Z_score or not. Defaults to False.
        trial_loss_area_specific (bool): If true population average per area else per all neurons from the session.

    Returns:
        _type_: _description_
    """
    if not trial_loss_area_specific:
        area *= 0
    feat_data = []
    feat_model = []
    for i, a in enumerate(area.unique()):
        if (area == a).sum() < 10:
            continue
        f_model = filt_model[..., area == a]
        f_data = filt_data[..., area == a]
        f_model, f_data = f_model.mean(2).T, f_data.mean(2).T
        if z_score:
            std = f_data[~f_data.isnan()].std()
            std[std < 0.0001] = 1
            mean = f_data[~f_data.isnan()].mean(0)
        else:
            mean, std = 0, 1
        f_model = (f_model - mean) / std
        f_data = (f_data - mean) / std
        feat_data.append(f_data)
        feat_model.append(f_model)
    if len(feat_data) == 0:
        return torch.tensor(feat_data), torch.tensor(feat_model)
    if filt_data_jaw is not None:
        sess = 0 if filt_model_jaw.shape[2] == 1 else session
        feat_model.append(filt_model_jaw[..., sess].T)
        feat_data.append(filt_data_jaw[..., session].T)
    feat_data = torch.cat(feat_data, dim=1)
    feat_model = torch.cat(feat_model, dim=1)
    return feat_data, feat_model


def mse_2d(x, y):
    """Creates a 2d matrix where the i,j element is the mse(x_i,y_i)

    Args:
        x (torch.tensor): input vector
        y (torch.tensor): output vector

    Returns:
        [torch.tensor]: 2d matrix of errors
    """
    x = x.T.repeat(y.shape[-1], 1, 1).permute(2, 1, 0)
    y = y.T.repeat(x.shape[-2], 1, 1).permute(2, 0, 1)

    return ((x - y) ** 2).mean(0)


def mse_2dv2(x, y):
    cost1 = torch.zeros(y.shape[0], x.shape[0], device=x.device)
    cost1[:, torch.arange(x.shape[0])] = (x * x).sum((1, 2))
    cost2 = torch.zeros(x.shape[0], y.shape[0], device=x.device)
    cost2[:, torch.arange(y.shape[0])] = (y * y).sum((1, 2))

    cost = cost1.T + cost2 - 2 * torch.einsum("ktn, ltn -> kl", x, y)
    return cost / x.shape[1] / x.shape[2]


def session_tensor(session, session_info, data, model, data_jaw, model_jaw):
    """The tensors from the model and data are referring to all the sessions, at the same time
    we do this for convenience BUT in some cases we need to split the tensor based on the session,
    e.g. trial-matching loss that's why we have developed the following function that based the session
    index and session_info we can split the big tensors.

    Args:
        session (int): Session index
        session_info (list): Information about the [trial_types, stims, trial_active, neuron_index]
        data (torch.tensor): data spikes tensor
        model (torch.tensor): model spikes tensor
        data_jaw (torch.tensor): data jaw trace
        model_jaw (torch.tensor): model jaw trace

    Returns:
        [data_spikes, model_spikes, ]: _description_
    """
    idx = session_info[-1][session]
    filt_model = model[:, :, idx]
    filt_data = data[:, :, idx]
    data_trials = torch.where(filt_data.sum((0, 2)).isnan() == False)[0]
    filt_data = filt_data[:, data_trials]
    if data_jaw is not None:
        data_jaw = data_jaw[:, data_trials]
    model_trials = torch.where(filt_model.sum((0, 2)).isnan() == False)[0]
    filt_model = filt_model[:, model_trials]
    if model_jaw is not None:
        model_jaw = model_jaw[:, model_trials]
    return filt_data, filt_model, data_jaw, model_jaw, idx


def pca_transform(signal1, signal2, n_components=10):
    # the implentation is basically the same as the sklearn pca with whiten
    mean = signal1.mean(0)
    u, s, v = torch.linalg.svd(torch.tensor(signal1 - mean))
    exp_var = s[:n_components] ** 2 / (u.shape[0] - 1)
    trans1 = torch.tensor(signal1 - mean) @ v[:n_components].T / exp_var**0.5
    trans2 = torch.tensor(signal2 - mean) @ v[:n_components].T / exp_var**0.5
    return trans1, trans2


def metrics(output, target):
    if target.dim() == 1:
        loss = torch.nn.BCELoss()(output, target)
    else:
        loss = torch.nn.CrossEntropyLoss()(output, target * 1.0)
    return loss


def make_template(
    model,
    data_spikes,
    session_info,
    jaw=None,
    num_trial_types=4,
    num_areas=6,
    remove_mean=True,
):
    dev = data_spikes.device
    neurons_per_area = 250
    area_index = model.rsnn.area_index
    T, K, N = data_spikes.shape
    sessions = [i for i in range(len(session_info[0]))]
    if jaw is not None:
        num_areas += 1
    template = torch.zeros(T * num_areas, num_trial_types, device=dev)
    for session in sessions:
        sess_data, _, jaw_data, _, idx = session_tensor(
            session, session_info, data_spikes, data_spikes, jaw, jaw
        )

        areas = area_index[idx]
        for tr_type in range(num_trial_types):
            tt = session_info[0][session] == tr_type
            for area in areas.unique():
                if (session_info[0][session] == tr_type).sum() == 0:
                    continue
                template[area * T : (area + 1) * T, tr_type] += (
                    sess_data[:, tt][..., areas == area].mean(1).sum(1)
                    / neurons_per_area
                )
            if jaw is not None and not (jaw_data[:, tt, session].mean(1).sum().isnan()):
                template[(num_areas - 1) * T : (num_areas) * T, tr_type] += jaw_data[
                    :, tt, session
                ].nanmean(1) / len(sessions)
    mean_area, std_area = [], []
    for area in range(num_areas):
        tmp = template[area * T : (area + 1) * T, :]
        if remove_mean:
            mean_area.append(tmp.mean())
            std_area.append(tmp.std())
        else:
            mean_area.append(0)
            std_area.append(1)
        template[area * T : (area + 1) * T, :] = (tmp.T - mean_area[area]).T / std_area[
            area
        ]
    return template, mean_area, std_area


def trial_match_template(
    model,
    data_spikes,
    session_info,
    model_spikes,
    jaw=None,
    data_jaw=None,
    consider_jaw=True,
    num_areas=6,
    num_trial_types=4,
    response=False,
    light=False,
    opto_area=None,
    period=None,
):
    data_spikes = model.filter_fun2(data_spikes)
    model_spikes = model.filter_fun2(model_spikes)
    if not consider_jaw:
        jaw, data_jaw = None, None
    if data_jaw is not None:
        jaw = model.filter_fun2(jaw)
        data_jaw = model.filter_fun2(data_jaw)
    data_template, mean_area, std_area = make_template(
        model,
        data_spikes,
        session_info,
        data_jaw,
        num_trial_types=num_trial_types,
        num_areas=num_areas,
    )
    T, K, N = model_spikes.shape
    neurons_per_area = 250
    area_index = model.rsnn.area_index
    if data_jaw is not None:
        num_areas += 1
    model_template = torch.zeros(num_areas * T, K, device=data_spikes.device)
    for area in range(num_areas - (data_jaw is not None)):
        tmp = model_spikes[:, :, area_index == area].sum(2) / neurons_per_area
        model_template[area * T : (area + 1) * T] = (
            tmp.T - mean_area[area]
        ).T / std_area[area]
    if data_jaw is not None:
        model_template[(area + 1) * T : (area + 2) * T] = jaw[:, :, 0]

    if light and not response:
        assert opto_area is not None or period is not None
        period = model.filter_fun2(period[:, None, None] * 1.0)[:, 0, 0] > 0
        new_time = []
        for i in range(num_areas):
            t = torch.ones(T) > 0 if i != opto_area else ~period
            new_time.append(t)
        new_time = torch.cat(new_time)
        model_template = model_template[new_time]
        data_template = data_template[new_time]
    if response:
        ton = int(
            0.2 / model.timestep / model.filter1.stride[0] / model.filter2.stride[0]
        )
        new_time = []
        for i in range(num_areas):
            t = torch.zeros(T) > 0
            t[-ton:] = True
            if i == opto_area and light:
                assert opto_area is not None or period is not None
                period1 = model.filter_fun2(period[:, None, None] * 1.0)[:, 0, 0] > 0
                t = t & (~period1)
            new_time.append(t)
        new_time = torch.cat(new_time)
        model_template = model_template[new_time]
        data_template = data_template[new_time]

    trial_types = torch.zeros(K)
    for k in range(K):
        trial_types[k] = (
            ((model_template[:, k] - data_template.T) ** 2).mean(1).argmin()
        )

    return trial_types


def most_common(vector):
    vector_unique = vector[~vector.isnan()].unique()
    output = torch.zeros(vector.shape[1])
    for i in range(vector.shape[1]):
        t = torch.tensor([(u == vector[:, i]).sum() for u in vector_unique])
        output[i] = torch.argmax(t)
    return output


def trial_type_active_quiet(
    model,
    model_spikes,
    data_spikes,
    session_info,
    jaw=None,
    data_jaw=None,
    method=0,
    return_full=False,
    response=False,
    consider_jaw=True,
    light=False,
    opto_area=None,
    period=None,
):
    num_trial_types = len(model.opt.trial_types)
    trial_type, active = model.return_hit_active1(
        model_spikes, data_spikes, data_jaw, jaw, session_info, consider_jaw
    )
    trial_type_nans = trial_type * 1.0
    trial_type_nans[trial_type_nans == 250] = torch.nan
    trial_types = most_common(trial_type_nans).cpu()
    if return_full:
        trial_types = trial_type
    aq_nans = active * 1.0
    aq_nans[aq_nans == 250] = torch.nan
    actives = most_common(aq_nans)
    if method != 0:
        trial_types = trial_match_template(
            model,
            data_spikes,
            session_info,
            model_spikes,
            jaw,
            data_jaw,
            consider_jaw,
            num_trial_types=num_trial_types,
            response=response,
            num_areas=len(model.opt.areas),
            light=light,
            opto_area=opto_area,
            period=period,
        )
    return trial_types, actives


def return_trial_type(
    model,
    data_spikes,
    model_spikes,
    data_jaw,
    model_jaw,
    session_info,
    stims,
    lick_detector=None,
):
    num_areas = len(model.opt.areas)
    num_trial_types = len(model.opt.trial_types)
    if lick_detector is None:
        trial_types = trial_match_template(
            model,
            data_spikes,
            session_info,
            model_spikes,
            model_jaw,
            data_jaw,
            num_trial_types=num_trial_types,
            num_areas=num_areas,
        )
    else:
        trial_types = lick_detector(model_jaw[:, :, 0].T).flatten() > 0.5
        trial_types = trial_types.clone() + (1 - stims) * 2
    trial_types_perc = torch.zeros(len(model.opt.trial_types))
    for i, j in enumerate(model.opt.trial_types):
        trial_types_perc[i] = (trial_types == j).sum() / trial_types.shape[0]
    return trial_types, trial_types_perc.cpu()


def trial_type_perc(session_info, num_trial_types=2):
    percentages = np.zeros(num_trial_types)
    total_trials = 0
    for sess in range(len(session_info[0])):
        p1 = np.unique(session_info[0][sess], return_counts=True)[1]
        total_trials += session_info[0][sess].shape[0]
        percentages += p1
    percentages /= total_trials
    return percentages


def trial_metric(
    filt_train,
    filt_test,
    filt_jaw_train,
    filt_jaw_test,
    session_info,
    model,
    measure="pear_corr",
    shuffle=False,
    modulated=None,
    seed=0,
):
    metric = []
    if measure == "pear_corr":
        measure = pear_corr
    elif measure == "explained":
        measure = explained
    for sess in range(len(session_info[0])):
        f_train, f_test, f_jaw_train, f_jaw_test, idx = session_tensor(
            sess,
            session_info,
            filt_train,
            filt_test,
            filt_jaw_train,
            filt_jaw_test,
        )
        if idx.sum() < 10:
            continue
        pop_train, pop_test = feature_pop_avg(
            f_train,
            f_test,
            f_jaw_train,
            f_jaw_test,
            model.rsnn.area_index[idx],
            sess,
            z_score=True,
            trial_loss_area_specific=model.opt.trial_loss_area_specific,
        )
        if len(pop_train) == 0:
            continue
        torch.manual_seed(seed)
        pop_train, pop_test = pop_train.T, pop_test.T
        keep = min(pop_train.shape[1], pop_test.shape[1])
        keep_train = torch.randperm(pop_train.shape[1])[:keep]
        keep_test = torch.randperm(pop_test.shape[1])[:keep]
        pop_train, pop_test = pop_train[:, keep_train], pop_test[:, keep_test]
        if not shuffle:
            cost = mse_2d(pop_train, pop_test)
            keepx, ytox = linear_sum_assignment(cost.detach().cpu().numpy())
            pop_train = pop_train[:, keepx]
            pop_test = pop_test[:, ytox]
        else:
            keep = min(pop_train.shape[1], pop_test.shape[1])
            keep_train = torch.randperm(pop_train.shape[1])[:keep]
            keep_test = torch.randperm(pop_test.shape[1])[:keep]
            pop_train, pop_test = pop_train[:, keep_train], pop_test[:, keep_test]
        metric.append(measure(pop_train.T, pop_test.T, dim=1))
    metric = np.concatenate(metric)
    return metric


def pear_corr(y, y_pred, eps=1e-8, dim=None):
    y, y_pred = y.cpu().numpy(), y_pred.cpu().numpy()
    if dim == 1:
        y_pred = (y_pred.T - y.mean(dim)).T
        y = (y.T - y.mean(dim)).T
    else:
        y_pred = y_pred - y.mean(dim)
        y = y - y.mean(dim)
    cov = (y * y_pred).sum(dim)
    y_sqr = (y * y).sum(dim)
    y_sqr_pred = (y_pred * y_pred).sum(dim)
    return cov / ((y_sqr * y_sqr_pred) ** 0.5 + eps)


def explained(y, y_pred, eps=1e-7, dim=None):
    y, y_pred = y.cpu().numpy(), y_pred.cpu().numpy()
    error = ((y - y_pred) ** 2).mean(dim)
    variance = y.var(dim) + eps
    r2 = 1 - (error / variance)
    r2[variance < eps] = np.nan
    return r2


def load_data(model):
    opt = model.opt
    dataset = TrialDataset(
        opt.datapath,
        model.sessions,
        model.areas,
        model.neuron_index,
        start=opt.start,
        stop=opt.stop,
        stim=opt.stim,
        trial_type=opt.trial_types,
        reaction_time_limits=opt.reaction_time_limits,
        timestep=opt.dt * 0.001,
        with_behaviour=opt.with_behaviour,
        trial_onset=opt.trial_onset,
    )
    dataset.to_torch()
    if dataset.with_behaviour:
        dataset.normalize_jaw()

    (
        data_spikes_train,
        _,
        data_jaw_train,
        session_info_train,
    ) = dataset.get_train_trial_type(
        train=1, device=opt.device, jaw_tongue=opt.jaw_tongue
    )
    (
        data_spikes_test,
        _,
        data_jaw_test,
        session_info_test,
    ) = dataset.get_train_trial_type(
        train=0, device=opt.device, jaw_tongue=opt.jaw_tongue
    )

    return (
        data_spikes_train,
        data_jaw_train,
        session_info_train,
        data_spikes_test,
        data_jaw_test,
        session_info_test,
    )
