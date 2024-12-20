# Overview
Tensorflow implementation of the experiment "Unsupervised Learning for UWMMSE Power Allocation" from the paper "Fully Distributed Online Training of Graph Neural Networks in Networked Systems". TODO: add link to the paper

## Dependencies

* **python>=3.6**
* **tensorflow>=1.14.0**: https://tensorflow.org
* **numpy**
* **matplotlib**

## Structure
* [datagen](https://github.com/RostyslavUA/Unrolled-WMMSE-distr/blob/master/datagen.py): Code to generate dataset. Generates A.pkl ( Geometric graph ), H.pkl ( Dictionary containing train_H and test_H ) and coordinates.pkl ( node position coordinates ).  Run as *python3 datagen.py* \[dataset ID\]. User chosen \[dataset ID\] will be used as the foldername to store dataset. Eg., to generate dataset with ID *set1*, run *python3 datagen.py set1*.
* [data](https://github.com/RostyslavUA/Unrolled-WMMSE-distr/tree/master/data): should contain your dataset in folder \[dataset ID\]. 
* [main](https://github.com/RostyslavUA/Unrolled-WMMSE-distr/blob/master/main.py): Main code for running the centralized training. Run as *python3 main.py* \[dataset ID\] \[exp ID\] \[mode\]. Eg., to train UWMMSE on dataset with ID set1, run *python3 main.py set1 uwmmse train*.
* [main_d](https://github.com/RostyslavUA/Unrolled-WMMSE-distr/blob/master/main.py): Main code for running the distributed training. Run as *python3 main_d.py* \[dataset ID\] \[exp ID\] \[mode\]. Eg., to train DUWMMSE on dataset with ID set1, run *python3 main_d.py set1 duwmmse train*.
* [model](https://github.com/RostyslavUA/Unrolled-WMMSE-distr/blob/master/model.py): Defines the DUWMMSE and UWMMSE models.
* [models](https://github.com/RostyslavUA/Unrolled-WMMSE-distr/tree/master/models): Stores trained models in a folder with same name as \[dataset ID\].
RostyslavUA

## Feedback
For questions and comments, feel free to contact [Rostyslav Olshevskyi](mailto:ro22@rice.edu).

## Citation
Please cite (TODO: add citation) in your work when using this library in your experiments.
