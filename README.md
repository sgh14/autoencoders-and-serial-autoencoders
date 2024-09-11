# autoencoders-and-serial-autoencoders


## Experiments

First create a conda environment:

```
$ conda env create -f experiments/environment.yml
```

then run

```
$ python -m experiments.swiss_roll.experiment
```

It is possible that the following additional installation is necessary to generate the plots with the desired font:

```
$ apt-get update
$ apt-get install -y cm-super dvipng texlive-latex-extra texlive-fonts-recommended
```