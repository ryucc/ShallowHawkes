# Shallow Learning for Hawkes Process

## Introduction

The Network Hawkes Process was proposed by Scott Lindermann [Link](http://jmlr.org/proceedings/papers/v32/linderman14.html).

I used a Convolutional NN like method on the same data to do some edge prediection, and got the impulse functions as a by-product.

## Usage

Under Construction.

The basic idea is to create a model class, give it a predictor, and set its parameters. See src/main.py

I inherit the variable names Scott used.
T: Total time.
K: Number of processes(nodes).
dt_max: The max delay a event could have influence on.
