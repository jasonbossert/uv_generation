# UV Generation 

This project investigate the efficiency of 118 nm photon production via a third-harmonic resonant process in xenon. Significant photon loss is observed, and possible mechanisms for this loss are proposed.

## Introduction

We are interested in experimentally generating 118 nm photos as part of a REMPI scheme to ionize OH molecules. We use a tripled Nd:YAG laser at 355 nm to pump a third-harmonic generation gas cell. This cell contains xenon, which has a large nonlinear susceptibility in the region of 118 nm, and argon, which phase matches the 355 nm light to the 118 nm light so that all generated 118 nm light constructively builds.

We would expect that as we increase the pressure of xenon, the macroscopic susceptibility would increase, and we would see an increase in the output 118 nm intensity. Indeed, we see this behavior, but only up to about 24 Torr partial pressure of xenon. At larger pressures, the intensity of 118 nm light decreases.

These notebooks look at one possible 118 nm loss mechanism, and then build a one and three dimensional models to better determine the accuracy of this physical mechanism.

## Linear Absorption

[Linear absorption](notebooks/vuv_gen_linear_absorption.ipynb) occurs when one photon at 118 nm is absorbed by a xenon atom. 

## One Dimensional Model

The [one dimensional model](notebooks/vuv_gen_1D_prop_model.ipynb) compares several differential equation models for 118 nm photon propagation, and fits the most likely model to extract a linear absorption parameter.

## Three Dimensional Model

The [three dimensional model](notebooks/vuv_gen_3D_prop_model.ipynb) uses Fourier optics to propagate a 355 nm beam and generate and lose 118 nm photons. The fully three-dimensional model has slight variations in predictions from the 1D model.
