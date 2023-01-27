#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

def latin_hypercube(n_pts, dim, perturb=True):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    if (perturb):
        # Add some perturbations within each box
        pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
        X += pert
    
    return X


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def euclidean_distance(vx, vy):
    return sum((y-x)**2 for x, y in zip(vx, vy)) ** 0.5


def centroid_x(x, dim):
    centroid_X = np.zeros((1, dim))
    for i in range(dim):
        for j in range(len(x)):
            centroid_X[:, i] += x[j, i]
    for i in range(dim):
        centroid_X[:, i] = centroid_X[:, i] / len(x)
    return centroid_X


