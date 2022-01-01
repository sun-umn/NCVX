<img src="./NCVX_logo_banner.png" alt="NCVX_LOGO" width="900"/>

# Introduction

**NCVX** (NonConVeX) is a user-friendly and scalable python software package targeting general nonsmooth NCVX problems with nonsmooth constraints. **NCVX** is being developed by [GLOVEX](https://glovex.umn.edu/) at the Department of Computer Science & Engineering, University of Minnesota, Twin Cities.

The initial release of NCVX contains the solver **PyGRANSO**, a PyTorch-enabled port of [GRANSO](http://www.timmitchell.com/software/GRANSO/) incorporating auto-differentiation, GPU acceleration, tensor input, and support for new QP solvers. As a highlight, **PyGRANSO** can solve deep learning problems with nontrivial constraints (e.g., constraints with neural networks), the first of its kind.

# Solver/Module

**PyGRANSO**: A PyTorch-enabled port of GRANSO with auto-differentiation [Documentation](https://ncvx.org/PyGRANSO) | [Repository](https://github.com/sun-umn/PyGRANSO)

## Citation

If you publish work that uses or refers to PyGRANSO, please cite the following two papers,
which respectively introduced PyGRANSO and GRANSO:

*[1] Buyun Liang, Tim Mitchell, and Ju Sun,
    NCVX: A User-Friendly and Scalable Package for Nonconvex
    Optimization in Machine Learning, arXiv preprint arXiv:2111.13984 (2021).*
    Available at https://arxiv.org/abs/2111.13984

*[2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton,
    A BFGS-SQP method for nonsmooth, nonconvex, constrained
    optimization and its evaluation using relative minimization
    profiles, Optimization Methods and Software, 32(1):148-181, 2017.*
    Available at https://dx.doi.org/10.1080/10556788.2016.1208749  

BibTex:

    @article{liang2021ncvx,
        title={{NCVX}: {A} User-Friendly and Scalable Package for Nonconvex 
        Optimization in Machine Learning}, 
        author={Buyun Liang, Tim Mitchell, and Ju Sun},
        year={2021},
        eprint={2111.13984},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    
    @article{curtis2017bfgssqp,
        title={A {BFGS-SQP} method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles},
        author={Frank E. Curtis, Tim Mitchell, and Michael L. Overton},
        journal={Optimization Methods and Software},
        volume={32},
        number={1},
        pages={148--181},
        year={2017},
        publisher={Taylor \& Francis}
    }
