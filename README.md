What is Right for Me is Not Yet Right for You:<br>A Dataset for Grounding Relative Directions via Multi-Task Learning
========

[Paper](https://www2.informatik.uni-hamburg.de/wtm/publications/2022/LKAWW22/GRID3D_IJCAI22.pdf) • [Video](toBeInserted) • [BibTex](toBeInserted) • [Dataset Download](https://www2.informatik.uni-hamburg.de/wtm/datasets2/grid-3d.zip)

This is the official repository associated with our [IJCAI-ECAI 2022](https://ijcai-22.org) paper, in which we present our novel VQA GRiD-3D (<u>**G**</u>rounding <u>**R**</u>elat<u>**i**</u>ve <u>**D**</u>irections in <u>**3D**</u>) dataset. The code was tested with python version X.X.X on macOS Monterey. 

If you find this work useful, please cite our [paper](https://www2alt.informatik.uni-hamburg.de/wtm/publications/2022/LKAWW22/index.php):

```
@InProceedings{lee_grid3d_2022,
  author       = "Lee, Jae Hee and Kerzel, Matthias and Ahrens, Kyra and Weber, Cornelius and Wermter, Stefan",
  title        = "What is Right for Me is Not Yet Right for You: A Dataset for Grounding Relative Directions via Multi-Task Learning",
  booktitle    = "International Joint Conference on Artificial Intelligence",
  year         = "2022",
  url          = "https://www2.informatik.uni-hamburg.de/wtm/publications/2022/LKAWW22/GRID3D_IJCAI22.pdf"
}
```

The experiments in our paper were conducted with the original versions of the MAC and FiLM models, which will be included as submodules in this repository.

## Environment Setup

First, clone the repository locally:
```
git clone https://github.com/knowledgetechnologyuhh/grid-3d.git
 ```

Create and activate an environment (we use [conda](https://docs.conda.io/en/latest/) here, but any other package manager does the job, too)
```
conda create -n grid3d_env python=X.X
conda activate grid3d_env
```

Install all packages listed in the `requirements.txt` file:
```
pip install -r requirements.txt
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

## Training and Evaluation

TBD


