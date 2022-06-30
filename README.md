Install the following packages with your favorite python package manager
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
or
```
conda env create -f environment.yml
```
Install the `vr` package used by FiLM:
```
cd grid3d/models/film
git checkout lee/optimize
python -m pip install -e .
```