# On Manifold Learning in Plato's Cave: Remarks on Manifold Learning and Physical Phenomena

Code for replicating the numerical experiments in

RR Lederman, B Toader. __On Manifold Learning in Plato's Cave: Remarks on Manifold Learning and Physical Phenomena__. _International Conference on Sampling Theory and Applications (SampTA)_, 2023.

[arXiv:2304.14248](https://arxiv.org/pdf/2304.14248)


## Requirements

The requirements are minimal (jax, numpy, matplotlib, jupyter). A conda environment can be created using the provided yml file:

```bash
conda env create -f environment.yml
conda activate jax_minimal
```


## Running the numerical experiments

The numerical experiments are in the ``notebooks`` directory. 

*  ``experiment_two_cameras.ipynb``: The main experiment of the paper, this notebook generates the figures in the main text (Fig. 2 and Fig. 3), the figure in Appendix A.2 showing the density of the points in the measurement space (Fig. 5), and the figure in Appendix A.3 showing the manifold embeddings and the densities from two cameras (Fig. 7). 

* ``experiment_angles.ipynb``: Generates the plots in Appendix A.1 showing the embedding of the angles (Fig. 4).

* ``experiment_top.ipynb``: Generates the plots in Appendix A.4 showing the embedding of the images from the top view (Fig. 9).

  
## Citation

If you found this code useful in academic work, please cite: ([arXiv link](https://arxiv.org/pdf/2304.14248))

```bibtex
@article{lederman2023manifold,
    title = {On {Manifold} {Learning} in {Plato}'s {Cave}: {Remarks} on {Manifold} {Learning} and {Physical} {Phenomena}},
    author = {Lederman, Roy R. and Toader, Bogdan},
    journal = {International Conference on Sampling Theory and Applications (SampTA 2023)},
    year = {2023},
}
```
