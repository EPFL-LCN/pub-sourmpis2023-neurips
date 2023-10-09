import torch
import matplotlib.pyplot as plt


def post_synaptic_weight(thetas, p_exc):
    n_e = int(thetas.shape[1] * p_exc)
    exc = thetas[:, :n_e]
    inh = thetas[:, n_e:]
    exc_sum = exc.sum(1)
    inh_sum = inh.sum(1)
    exc = torch.where(inh_sum > exc_sum, exc.T, exc.T * inh_sum / exc_sum).T
    inh = torch.where(inh_sum < exc_sum, inh.T, inh.T * exc_sum / inh_sum).T
    return torch.cat((exc, inh), dim=1)


def divide_max_eig(thetas, p_exc):
    n_e = int(thetas.shape[0] * p_exc)
    sign = torch.ones_like(thetas)
    sign[:, n_e:] *= -1
    max_eig = torch.linalg.eigvals(thetas * sign).abs().max()
    thetas /= max_eig
    return thetas


def restricted_inter_inh_thetas_init(n_neurons, p_exc, area_index, eps=0):
    n_e = int(n_neurons * p_exc)
    thetas = torch.zeros(n_neurons, n_neurons)
    intra_area_mask = torch.zeros(n_neurons, n_neurons)
    n_areas = area_index.max() + 1
    for area in range(n_areas):
        area_vec = area_index == area
        area_mask = ((area_vec) * 1.0)[:, None] @ ((area_vec) * 1.0)[None]
        thetas[area_mask > 0] = torch.rand(area_vec.sum(), area_vec.sum()).flatten()
        intra_area_mask[area_mask > 0] = 1
    inter_ee = intra_area_mask == 0
    inter_ee[:, n_e:] = 0
    thetas = thetas + eps * inter_ee * torch.rand_like(thetas)
    return thetas


if __name__ == "__main__":
    n_neurons = 1500
    n_areas = 6
    p_ei = 0.8
    area_index = torch.zeros(n_neurons)
    n_exc = int(n_neurons * p_ei)
    n_inh = n_neurons - n_exc
    for i in range(n_neurons):
        exc_per_area = n_exc // n_areas
        inh_per_area = n_inh // n_areas
        if i < n_exc:
            area_index[i] = i // exc_per_area
        else:
            area_index[i] = (i - n_exc) // inh_per_area

    thetas = torch.zeros(n_neurons, n_neurons)
    sign = torch.ones_like(thetas)
    sign[:, n_exc:] *= -1
    intra_area_mask = torch.zeros_like(thetas)
    for area in range(n_areas):
        area_ind = area_index == area
        area_mask = ((area_ind) * 1.0)[:, None] @ ((area_ind) * 1.0)[None]
        intra_area_mask[area_mask > 0] = 1
        thetas_minor = torch.rand(area_ind.sum(), area_ind.sum())
        thetas_minor = post_synaptic_weight(thetas_minor, p_ei)
        thetas_minor = divide_max_eig(thetas_minor, p_ei)
        thetas[area_mask > 0] = thetas_minor.flatten()
    eig, vec = torch.linalg.eig(thetas * sign)
    plt.scatter(eig.real, eig.imag)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    thetas_all = torch.rand(n_neurons, n_neurons)
    thetas_all = post_synaptic_weight(thetas_all, p_ei)
    thetas_all = divide_max_eig(thetas_all, p_ei)
    eig, vec = torch.linalg.eig(thetas_all * sign)
    ax.scatter(
        eig.real,
        eig.imag,
        label=f"full,  ratio of max imag to max eig {1/eig.imag.max()}",
    )

    inter_ee = intra_area_mask == 0
    inter_ee[:, n_exc:] = 0
    for i in range(5):
        eps = 0.001 * i
        th = thetas + eps * inter_ee * torch.rand_like(thetas)
        th = post_synaptic_weight(th, p_ei)
        th = divide_max_eig(th, p_ei)
        eig, vec = torch.linalg.eig(th * sign)
        ax.scatter(
            eig.real,
            eig.imag,
            label=f"no inter, inh eps = {eps}, ratio of max imag to max eig {1/eig.imag.max():.2f}",
        )
    ax.legend()
    fig.savefig("effect_of_inter_area_e_e.png")
    plt.show()

    plt.figure()
    plt.pcolormesh(intra_area_mask)
    plt.figure()
    plt.pcolormesh(thetas)
    plt.show()
