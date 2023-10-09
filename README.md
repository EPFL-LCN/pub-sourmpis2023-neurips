# Trial matching code

This is the code for the publication:
C. Sourmpis, C. Petersen, W. Gerstner & G. Bellec
[*Trial matching: capturing variability with data-constrained spiking neural networks*](https://neurips.cc/virtual/2023/poster/71974), accepted at NeurIPS 2023.

Contact:
[christos.sourmpis@epfl.ch](mailto:christos.sourmpis@epfl.ch)

The code is sufficient in order to generate all the figures of the paper, and the code to generate the figures of the main text is contained in notebooks in the folder Figures.

In order to run the code you need to do 2 main steps before:
1. Install this code as a module in python
2. Download the data (for the recorded data) and generate the data (for the artificial dataset)

Now you are ready to run the code to generate the figures.

Training models will require a little bit better understanding of the code, HOWEVER you can train the main model with the following command, and you could start exploring the parameters, by changing the options in the file configs/main_model/opt.json:

```bash
python3 infopath/train.py --config=main_model
```
The previous command is suppoded to be run on GPU. Be careful that this training will require GPU RAM of at least 40GB. If you want to run it with CPU, you can change the field "device" in the configs/main_model/opt.json.

## Installation
We suggest installing the code with conda and you can do this with the following way:

```bash
conda create --name trial-match python=3.9.5
pip install -e .
```
Now you should be able to run the code.

## Data
### Recorded data
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

### Artificial data
For the artificial data just run the command:
```bash
python datasets/pseudodata.py
```
This will generate the data for the figure 2. In order to get the data for the supplementary you need to modify the pseudodata.py file.



### Notes
For the Figure 4C you might notice that the UMAP is not the same as the one with the paper, this happens because we changed the function that generates the input spikes for readability. However, you can appreciate that the message of the main paper remains the same.