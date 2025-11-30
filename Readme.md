# Interactive Segmentation

This repository contains the code for **interactive segmentation** based on object creation, feature-space clustering, and flexible segmentation routines. It was originally developed as part of an image-analysis workflow, where individual objects are detected, described by shape features, clustered, and then segmented in a reproducible way.

The original archived version of this software is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14251450.svg)](https://doi.org/10.5281/zenodo.14251450)

---

## Features

- **Object creation**  
  Functions to detect target objects, crop them, and build an object set with extracted shape features.

- **Feature-space clustering**  
  Automatic clustering of objects in feature space (number of clusters is determined during clustering).

- **Segmentation**  
  Modular segmentation routines that can be extended with additional methods to evaluate performance on your objects.

---
# Data

The original archived data file `input.npy` used with this code is available on Zenodo:

- DOI: https://doi.org/10.5281/zenodo.14251450

To run the full example:

1. Download `input.npy` from Zenodo.
2. Create a folder `Workspace_data` at the project root.
3. Place `input.npy` inside `Workspace_data/`.
4. Create an empty `Raw_Figs` folder for generated figures.

You may also create your own `input.npy` with the same structure to adapt the pipeline to your data.


##
How to run
## Quickstart

```bash
git clone https://github.com/hamideh-h/interactive-segmentation.git
cd interactive-segmentation
pip install -r requirements.txt

# Prepare data
mkdir Workspace_data Raw_Figs
# put input.npy into Workspace_data/

# Run pipeline
python src/main.py

## Repository Layout
```
```text
src/
  main.py                  # entry point
  Segmentation.py          # segmentation functions
  FeatureSpace_Clustering.py
  Object.py
  objectset_creation.py
data/
  README.md                # explains where to get input.npy
requirements.txt           # Python dependencies

```

