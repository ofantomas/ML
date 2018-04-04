# SVM documentation

This folder contains source code for solving SVM optimization problem via Subgradient Descend, PEGASOS and quadratic programming problem solver. Kernel SVM is implementd via quadratic programming problem solver only. Also contains .ipynb file with different experiments on synthetic data.

__optimization.py__

SubGradient Descend Classifier, Stochastic SubGDC, PEGASOSClassifier compatible with any oracle, mathcing specification from oracles.py

__oracles.py__

Oracle for BinaryHinge and BinaryLogistic loss-functions

__svm.py__

SVM and kernel SVM via quadratic programming problem solver.
