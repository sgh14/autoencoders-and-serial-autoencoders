# autoencoders-and-serial-autoencoders

## Experiments

First create a conda environment:

```bash
$ conda env create -f experiments/environment.yml
```

then run

```bash
$ python -m experiments.swiss_roll.experiment
```

To use the same style sheet for matplotlib, LaTex must be installed as well:

```bash
$ sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```