# The MegaAttitude Project

**Authors:** Aaron Steven White and Kyle Rawlins

**Contact:** {aswhite,kgr}@jhu.edu

## Overview

This package implements White & Rawlins' (2016) computational model of s(emantic)-selection. Included in the `bin/` directory is a dataset consisting of ordinal acceptability judgments for ~1000 clause-embedding verbs of English—with 50 surface-syntactic frames per verb, 5 observations per verb-frame pair.  For a detailed description of the model and the data set as well as details of the item construction and collection methods, please see the following paper:

White, A. S. & K. Rawlins. 2016. [A computational model of S-selection](http://aswhite.net/media/papers/white_computational_2016_salt.pdf). In M. Moroney, C-R. Little, J. Collard & D. Burgdorf (eds.), *Semantics and Linguistic Theory* 26, 641-663. Ithaca, NY: CLC Publications.

If you make use of this data set in a presentation or publication, we ask that you please cite this paper.

## Installation

To install, simply use `pip`.

```bash
sudo pip install git+git://github.com/aaronstevenwhite/CHILDESPy.git
```

You can also clone and use the included `setup.py`.

```bash
git clone https://github.com/aaronstevenwhite/CHILDESPy
cd CHILDESPy
python setup.py install
```

Please note that this package is under active development and will undergo possibly major changes.
