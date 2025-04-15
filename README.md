# TransFusion: Generating Long, High Fidelity Time Series using Diffusion Models with Transformer

This repo contains the code for TransFusion paper (https://doi.org/10.1016/j.mlwa.2025.100652).

First, install the docker container by running the shell `build.sh`, after that run the environment by using `run-shell.sh` shell.

```terminal
chmod +x build.sh
./build.sh
chmod +x run-shell.sh
./run-shell.sh
```

## Training

To train TransFusion, run the following command inside the docker container.

Dataset Option: `sine, stock, air, energy`

To download the dataset (stock, air and energy), please refer to the supplementary document in this repo. Create a folder called `data` and paste the datasets there. If you want to change the code of dataset please refer to the `data_make.py` file.


```terminal
python3 train.py --dataset_name sine --seq_len 100 --batch_size 256
```

### Generated samples from the runtime can be found in the `saved_files` directory after running the experiments.

For Long-Sequence Discriminative score metric, we need both original data and synthetic data as `torch tensor`.

```python
from long_discriminative_score import long_discriminative_score_metrics


long_discriminative_score_metrics(original_data, synthetic_data)
```

For Long-Sequence Predictive score metric, we need both original data and synthetic data as `torch tensor`.

```python
from long_predictive_score import long_predictive_score_metrics

long_predictive_score_metrics(original_data, synthetic_data)
```

### Citation

Please refer to our work if you use any parts of the code:

```terminal
@article{sikdertransfusion2025,
	title = {TransFusion: Generating long, high fidelity time series using diffusion models with transformers},
	author={Sikder, Md Fahim and Ramachandranpillai, Resmi and Heintz, Fredrik},
	journal = {Machine Learning with Applications},
	volume = {20},
	pages = {100652},
	year = {2025},
	issn = {2666-8270},
	doi = {https://doi.org/10.1016/j.mlwa.2025.100652},
	url = {https://www.sciencedirect.com/science/article/pii/S2666827025000350}}
```