# The MegaAttitude dataset

**Authors:** Aaron Steven White and Kyle Rawlins

**Contact:** {aswhite,kgr}@jhu.edu

**Version:** 1.0

**Release date:** Oct 30, 2016

## Overview

This data set consists of ordinal acceptability judgments for ~1000 clause-embedding verbs of English — with 50 surface-syntactic frames per verb, 5 observations per verb-frame pair.  The data was collected on Amazon's Mechanical Turk using [Turktools](http://turktools.net/).  For a detailed description of the data set, the item construction and collection methods, and discussion of how to use a data set on this scale to address questions in linguistic theory, please see the following paper:

White, A. S. & K. Rawlins. 2016. [A computational model of S-selection](http://aswhite.net/papers/white_computational_2016_salt.pdf). In M. Moroney, C-R. Little, J. Collard & D. Burgdorf (eds.), *Semantics and Linguistic Theory* 26, 641-663. Ithaca, NY: CLC Publications.

If you make use of this data set in a presentation or publication, we ask that you please cite this paper.

## Version history

1.0: first public release, Oct 30, 2016.

## Description

| **Column**        | **Description**                                                                           | **Values**          |
|-------------------|-------------------------------------------------------------------------------------------|---------------------|
| participant       | anonymous integer identifier for participant that provided the response                   | 0...728             |
| list              | integer identifier for list participant was responding to                                 | 0...999             |
| presentationorder | relative position of item in list                                                         | 1...50              |
| verb              | clause-embedding verb found in the item                                                   | see paper           |
| frame             | clausal complement found in the item                                                      | see paper           |
| voice             | voice found in the item                                                                   | `active`, `passive` |
| response          | ordinal scale acceptability response                                                      | 1...7               |
| agreement         | how much participant agreed with other participants (see White & Rawlins, in prep)        | \[-3.96, 2.32\]     |
| nativeenglish     | whether the participant reported speaking American English natively                       | `True`, `False`     |
| exclude           | whether the participant should be excluded based on native language or agreement (<=-2.5) | `True`, `False`     |

## Notes

* Only participants for which `exclude==True` are included in the analysis in White & Rawlins 2016. The full exclusion procedure is laid out in a paper in preparation.
* A javascript error produced 10 NA values for `response`, none of which affect the same verb-frame pair.
