# Trial matching code

This is the code for the publication:
C. Sourmpis, C. Petersen, W. Gerstner & G. Bellec
[*Trial matching: capturing variability with data-constrained spiking neural networks*](https://neurips.cc/virtual/2023/poster/71974), accepted at NeurIPS 2023.

Contact:
[christos.sourmpis@epfl.ch](mailto:christos.sourmpis@epfl.ch)


## Glossary
1) [Installation](#Installation)
2) Download recorded data from Esmaeli et al. 2021
3) Generate simpler artificial data
4) Load pre-trained models
5) Code snippet for computing the trial-matching loss function
6) Generate paper figures
7) [Training the RSNN](#Training_the_RSNN_model)

## Installation
We suggest installing the code with conda and you can do this with the following way:

```bash
conda create --name trial-match python=3.9.5
pip install -e .
```
Now you should be able to run the code.

## Download recorded data
Be aware that the full data are ~55GB, but in the end we will use only ~500MB.

In order to use the recorded data you either can do it manually 
1. download the data from [here](https://zenodo.org/record/4720013), 
2. unzip 
3. from Electrophysilogy folder keep the spike_data_v9.mat 
4. put the spike_data_v9.mat in the datasets folder 

or run the following commands:

```bash
wget https://zenodo.org/record/4720013/files/Esmaeili_data_code.zip -P datasets
unzip datasets/Esmaeili_data_code.zip -d tmp
mv tmp/Electrophysiology/Data/spikeData_v9.mat datasets/spikeData_v9.mat
rm -r tmp
```

## Generate Artificial data
For the artificial data just run the command:
```bash
python datasets/pseudodata.py
```
This will generate the data for the figure 2. In order to get the data for the supplementary you need to modify the pseudodata.py file.

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

For instance the recurrent weights of the model can be obtained with:
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



## Generate figures from a pre-trained model

The code is sufficient in order to generate all the figures of the paper, in the folder `Figures` one can find the paper figures and notebooks to generate all the panels.

## Training the RSNN model

Training models will require a little bit better understanding of the code, HOWEVER you can train the main model with the following command, and you could start exploring the parameters, by changing the options in the file configs/main_model/opt.json:

```bash
python3 infopath/train.py --config=main_model
```
The previous command is suppoded to be run on GPU. Be careful that this training will require GPU RAM of at least 40GB. If you want to run it with CPU, you can change the field "device" in the configs/main_model/opt.json.

### Notes
For the Figure 4C you might notice that the UMAP is not the same as the one with the paper, this happens because we changed the function that generates the input spikes for readability. However, you can appreciate that the message of the main paper remains the same.
