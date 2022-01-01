<img src="./NCVX_logo_banner.png" alt="NCVX_LOGO" width="900"/>

# Introduction

**NCVX** (NonConVeX) is a user-friendly and scalable python software package targeting general nonsmooth NCVX problems with nonsmooth constraints. **NCVX** is being developed by [GLOVEX](https://glovex.umn.edu/) at the Department of Computer Science & Engineering, University of Minnesota, Twin Cities.

The initial release of NCVX contains the solver **PyGRANSO**, a PyTorch-enabled port of [GRANSO](http://www.timmitchell.com/software/GRANSO/) incorporating auto-differentiation, GPU acceleration, tensor input, and support for new QP solvers. As a highlight, **PyGRANSO** can solve deep learning problems with nontrivial constraints (e.g., constraints with neural networks), the first of its kind.

# Solver/Module

**PyGRANSO**: A PyTorch-enabled port of GRANSO with auto-differentiation [Documentation](https://ncvx.org/PyGRANSO) | [Repository](https://github.com/sun-umn/PyGRANSO)
