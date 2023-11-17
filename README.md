[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10006599.svg)](https://doi.org/10.5281/zenodo.10006599)

# Trial matching code

This is the code for the publication:
C. Sourmpis, C. Petersen, W. Gerstner & G. Bellec
[*Trial matching: capturing variability with data-constrained spiking neural networks*](https://neurips.cc/virtual/2023/poster/71974), accepted at NeurIPS 2023.

Contact:
[christos.sourmpis@epfl.ch](mailto:christos.sourmpis@epfl.ch)


## Glossary
1) [Installation](#Installation)
2) [Generate simpler artificial data](#generate-artificial-data)
3) [Load pre-trained models](#pre-trained-models)
4) [Code snippet for computing the trial-matching loss function](#compute-the-trial-matching-loss-function)
5) [Download recorded data from Esmaeli et al. 2021](#download-recorded-data)
6) [Generate paper figures](#generate-figures-from-a-pre-trained-model)
7) [Training the RSNN](#training-the-rsnn-model)

## Installation
We suggest installing the code with conda and you can do this in the following way:

```bash
conda create --name trial-match python=3.9.5
conda activate trial-match
pip install -e .
```
Now you should be able to run the code.

## Generate Artificial data
For the artificial data just run the command:
```bash
python datasets/pseudodata.py
```
This will generate the data for Figure 2 and the data for some of the supplementary Figures.

## Pre-trained models 

One can load a pre-trained model as follows.

```python
from infopath.model_loader import load_model_and_optimizer
from infopath.config import load_training_opt

log_path = "log_dir/trained_models/main_model/"
opt = load_training_opt(log_path)
opt.log_path = log_path
opt.device = "cpu"
model = load_model_and_optimizer(opt, reload=True, last_best="last")[0]
```

For instance, the recurrent weights of the model can be obtained with the following:
```python
model.rsnn._w_rec # shape: 2 x 1500 x 1500
```

To simulate a raster of 400 trials from the model one can do:
```python
with torch.no_grad():
    stims = torch.randint(2, size=(400,)) # binary vector of conditions (absence or presence of whisker stimulation)
    spikes, voltages, jaw, state = model(stims) # generation of the input spikes and simulation of the RSNN
```

## Compute the trial matching loss-function

Calculate the trial-matching loss with the hard matching (Hungarian Algorithm)

Args:


* filt_data_spikes (torch.tensor): $\mathcal{T}_{trial}(z^\mathcal{D})$, with dimension: K x T

* filt_model_spikes (torch.tensor): $\mathcal{T}_{trial}(z)$, with dimension K'  x T

```python
def hard_trial_matching_loss(filt_data_spikes, filt_model_spikes):
    # Subsample the biggest tensor, so both data and model have the same #trials
    min_trials = min(filt_model_spikes.shape[0], filt_data_spikes.shape[0])
    filt_data_spikes = filt_data_spikes[:min_trials] # shape: K x T (assuming K = min(K,K'))
    filt_model_spikes = filt_model_spikes[:min_trials] # shape: K x T
    with torch.no_grad():
        cost = mse_2d(filt_model_spikes.T, filt_data_spikes.T) # shape: K x K 
        keepx, ytox = linear_sum_assignment(cost.detach().cpu().numpy()) # keepx and ytox are trial indices
    return torch.nn.MSELoss()(filt_model_spikes[keepx], filt_data_spikes[ytox])
```

The function above is in the `infopath/losses.py` file.

You can explore the loss function in a simple demo in [trial_matching_loss_demo.ipynb](trial_matching_loss_demo.ipynb).

## Download recorded data
Be aware that the full data is ~55GB, but we will use only ~3GB in the end.

In order to use the recorded data you can do it manually 
1. download the data from [here](https://zenodo.org/record/4720013), 
2. unzip 
3. from the Electrophysiology folder keep the spike_data_v9.mat 
4. put the spike_data_v9.mat in the datasets folder 

or run the following commands:

```bash
wget https://zenodo.org/record/4720013/files/Esmaeili_data_code.zip -P datasets
unzip datasets/Esmaeili_data_code.zip -d tmp
mv tmp/Electrophysiology/Data/spikeData_v9.mat datasets/spikeData_v9.mat
rm -r tmp
```

## Generate figures from a pre-trained model

The code is sufficient in order to generate all the figures of the paper, in the folder `Figures` one can find the paper figures and notebooks to generate all the panels.

## Training the RSNN model

Training models will require a little bit better understanding of the code. However, you can train the main model with the following command, and you can start exploring the parameters, by changing the options in the file configs/main_model/opt.json:

```bash
python3 infopath/train.py --config=main_model
```
The previous command is supposed to be run on GPU. Be careful that this training will require GPU RAM of at least 40GB. If you want to run it with CPU, you can change the field "device" in the configs/main_model/opt.json.

### Notes
For the Figure 4C you might notice that the UMAP is not the same as the one with the paper, this happens because we changed the function that generates the input spikes for readability. However, you can appreciate that the message of the main paper remains the same.

### Citation 
You can cite us with the following bibtex:
```latex
@article{
    sourmpis2023trialmatching,
    title     = {Trial matching: capturing variability with data-constrained spiking neural networks},
    author    = {Sourmpis, Christos and Petersen, Carl C H and Gerstner, Wulfram and Bellec, Guillaume},
    journal   = {Advances in Neural Information Processing Systems},
    year      = {2023}
}
```
and here is the citation for the data:
```latex
@article{
    esmaeili2021rapid,
    title     = {Rapid suppression and sustained activation of distinct cortical regions for a delayed sensory-triggered motor response},
    author    = {Esmaeili, Vahid and Tamura, Keita and Muscinelli, Samuel P and Modirshanechi, Alireza and Boscaglia, Marta and Lee, Ashley B and Oryshchuk, Anastasiia and Foustoukos, Georgios and Liu, Yanqi and Crochet, Sylvain and Petersen, Carl C.H.},
    journal   = {Neuron},
    year      = {2021},
}
```
