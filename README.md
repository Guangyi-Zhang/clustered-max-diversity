# Maximizing diversity over clustered data

This repo hosts the code for the paper "G. Zhang AND A. Gionis, Maximizing diversity over clustered data, 2020".

The main algorithms are in `intra.py`, and metrics are run by Jupyter notebooks that start with `metric-`.
The repo adopts [sacred](https://github.com/IDSIA/sacred) and [incense](https://github.com/JarnoRFB/incense) to manage experiment results.
Therefore you will need a Mongo DB server to run experiments.

External realistic data can be downloaded from links below:

* https://grouplens.org/datasets/movielens/
* https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
* https://www.aminer.cn/data#Academic-Social-Network
