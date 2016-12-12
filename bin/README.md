# Experiments

This directory contains scripts for fitting different models to the data found in the `data/` directory. Each script defines a subclass of the Experiment class exposed at the top level of the `projectionmodel` package.

For instance, after installing the package, you can invoke `run_ordinal_factorizer.py` while in `bin/` to run the ordinal factorizer experiment.

```bash
git clone https://github.com/aaronstevenwhite/MegaAttitudeProject
cd MegaAttitudeProject
python setup.py install
cd bin
python run_ordinal_factorizer.py
```

You may wish to use python's `-i` option to probe the model fits, since by default, this script does not write anything.
