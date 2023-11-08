from matplotlib import pyplot as plt
from infopath.config import get_opt, load_training_opt, save_opt
import os
import pandas as pd
import json
import matplotlib as mpl

new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)


def create_configs():
    opt = get_opt(os.path.join("configs", "large_dt_art_data"))
    j = 0
    for dt in [20, 200]:
        for syn_delay in [2]:
            for tau_mem in [10]:
                path = f"datasets/PseudoData_v17_delay{dt}_onesession"
                opt.datapath = path
                opt.rec_groups = 2
                opt.n_delay = syn_delay
                opt.inter_delay = syn_delay * 2
                opt.tau_list = [tau_mem for _ in opt.tau_list]
                opt.areas = ["area1", "area2"]
                if not os.path.exists(f"configs/config{j}"):
                    os.mkdir(f"configs/config{j}")
                save_opt(f"configs/config{j}", opt)
                j += 1


create_configs()
