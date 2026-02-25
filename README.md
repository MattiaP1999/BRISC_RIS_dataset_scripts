# BRISC_RIS_dataset_scripts

Scripts to process the BRISC dataset introduced in the paper:

BRISC: A Dataset of Channel Measurements at 5 GHz With a Reflective
Intelligent Surface (arXiv: 2602.21102)

This repository allows reproduction of the results presented in the
paper and provides utilities for dataset preprocessing and model-based
estimation.

Repository Structure

-   `my_lib.py`: Custom functions for BRISC data preprocessing and feature extraction.

-   `nn_estimation.py`: Neural network-based channel estimation and regression experiments.

-   `rf_estimation.py`: Random forest-based estimation experiments.

-   `linear_model_estimation.py` Linear models experiments (LMB and LM).

Dataset Access

The BRISC dataset is publicly available on Zenodo:
https://zenodo.org/records/18714621

# Citation

If you use this dataset or the provided scripts, please cite:

@misc{piana2026briscdatasetchannelmeasurements,
      title={BRISC: A Dataset of Channel Measurements at 5 GHz With a Reflective Intelligent Surface},

      author={Mattia Piana and Giovanni Angelo Alghisi and Anna Valeria Guglielmi and Giovanni Perin and Francesco Gringoli and Stefano Tomasin},

      year={2026},

      eprint={2602.21102},

      archivePrefix={arXiv},

      primaryClass={eess.SP},
      
      url={https://arxiv.org/abs/2602.21102}, 
}


