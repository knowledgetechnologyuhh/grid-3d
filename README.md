What is Right for Me is Not Yet Right for You:<br>A Dataset for Grounding Relative Directions via Multi-Task Learning
========

<!-- [Paper](https://arxiv.org/abs/2205.02671) • [Video](toBeInserted) • [BibTex](toBeInserted) • [Dataset Download](https://www2.informatik.uni-hamburg.de/wtm/datasets2/grid-3d.zip) -->
[Paper](https://arxiv.org/abs/2205.02671) • [Dataset Download](https://www2.informatik.uni-hamburg.de/wtm/datasets2/grid-3d.zip)

This is the official repository associated with our [IJCAI-ECAI 2022](https://ijcai-22.org) paper, in which we present our novel VQA GRiD-3D (<u>**G**</u>rounding <u>**R**</u>elat<u>**i**</u>ve <u>**D**</u>irections in <u>**3D**</u>) dataset. The code was tested with python version 3.9 on Ubuntu 20.04. 

If you find this work useful, please cite our [paper](https://www2alt.informatik.uni-hamburg.de/wtm/publications/2022/LKAWW22/index.php):

```
@InProceedings{lee_grid3d_2022,
  author       = "Lee, Jae Hee and Kerzel, Matthias and Ahrens, Kyra and Weber, Cornelius and Wermter, Stefan",
  title        = "What is Right for Me is Not Yet Right for You: A Dataset for Grounding Relative Directions via Multi-Task Learning",
  booktitle    = "International Joint Conference on Artificial Intelligence",
  year         = "2022",
  url          = "https://arxiv.org/abs/2205.02671"
}
```

The experiments in our paper were conducted with the original versions of the MAC and FiLM models, which will be included as submodules in this repository.

## Environment Setup

First, clone the repository locally:
```
git clone https://github.com/knowledgetechnologyuhh/grid-3d.git
```
Then go to the root of the folder and install all necessary Python packages, conveniently using [Anaconda](https://docs.conda.io/en/latest/):
```
conda env create -f environment.yml
conda activate grid3d_env
```
or manually by instally the following packages:
```
ipdb 
munch
python=3.9 
pytorch 
pytorch-lightning=1.5.10 
termcolor 
torchmetrics 
torchvision 
yacs 
```
Now, download the submodules for the VQA models FiLM and MAC and install the `grid3d` package and the `vr` package (used by the FiLM model):
```
git submodule update --init --recursive
python -m pip install -e .
cd grid3d/models/film
git checkout lee/optimize
python -m pip install -e .
```

Next step is to download and extract the dataset, which we explain in the following section.

## GRiD-3D Dataset

![Overview](images/dataset_overview.png)

You can download the dataset by clicking [here](https://www2.informatik.uni-hamburg.de/wtm/datasets2/grid-3d.zip) or by running the following commands in your terminal:

#### Ubuntu / Linux:
```
wget https://www2.informatik.uni-hamburg.de/wtm/datasets2/grid-3d.zip
```

#### OS X:

```
curl -O https://www2.informatik.uni-hamburg.de/wtm/datasets2/grid-3d.zip
```

The default path for the dataset is set to `/data/grid3d/`, but you can change the path by modifying the corresponding entry in the `train_mac.py` and `train_film.py`:
```
cfg.DATASET.PATH = "path/to/the/dataset/folder"
```
## Running an Experiment
Run
`python grid3d/train_film.py` or `python grid3d/train_mac.py` to the replicate the experiments on the GRiD-3D dataset with all six tasks.
To run experiments on selected tasks, set the size of the corresponding task in `train_film.py` or `train_mac.py` to 0, e.g, to omit the **Link Prediction** task: 
```
cfg.TASK_SIZES.existence_prediction = 59053
cfg.TASK_SIZES.orientation_prediction = 26344
cfg.TASK_SIZES.link_prediction = 0 # 40770
cfg.TASK_SIZES.relation_prediction = 69800
cfg.TASK_SIZES.counting = 92904
cfg.TASK_SIZES.triple_classification = 166603
```