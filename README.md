# Symplectic Taylor Neural Networks
* Paper: https://arxiv.org/abs/2005.04986


## Summary
We propose an effective and light-weighted learning algorithm, Symplectic Taylor Neural Networks (Taylor-nets), to conduct continuous, long-term predictions of a complex Hamiltonian dynamic system based on sparse, short-term observations. At the heart of our algorithm is a novel neural network architecture consisting of two sub-networks. Both are embedded with terms in the form of Taylor series expansion that are designed with a symmetric structure. The key mechanism underpinning our infrastructure is the strong expressiveness and special symmetric property of the Taylor series expansion, which can inherently accommodate the numerical fitting process of the spatial derivatives of the Hamiltonian as well as preserve its symplectic structure. We further incorporate a fourth-order symplectic integrator in conjunction with neural ODEs’ framework into our Taylor-net architecture to learn the continuous time evolution of the target systems while preserving their symplectic structures simultaneously. 


![](https://github.com/ytong6/Taylor-net/blob/master/Figures/net2.png)

![](https://github.com/ytong6/Taylor-net/blob/master/Figures/net.png)


## Usage
We demonstrated the efficacy of our Tayler-net in predicting a broad spectrum of Hamiltonian dynamic systems, including the pendulum, the Lotka-Volterra, the Kepler, and the Hénon–Heiles systems.

In order to train a Taylor-net:
* Pendulum: `python3 Pendulum/Pendulum.py`
* Lotka-Volterra: `python3 Lotka_Volterra/LV.py`
* Hénon-Heiles: `python3 Henon_Heiles/Henon_Heiles.py`

## Problem setups

| Problems      | Pendulum      | Lotka-Volterra| Hénon-Heiles | Kepler |
| ------------- | ------------- | ------------- |------------- |------------- |
| Training period | 0.01  | 0.01  | 0.01  | 0.01 |
| Predicting period | 20$\pi$  | 20$\pi$  | 10  | 20$\pi$  |
| Sample size  | 15  | 25  | 25  | 25  |
| Epoch  | 100  | 150  | 100  | 50  |
| Learning rate  | 0.002  | 0.003  | 0.001  | 0.001  |
| $step\_size$  | 10  | 10  | 10  | 10  |
| $\gamma$  | 0.8  | 0.8  | 0.8  | 0.8  |
| $M$*  | 8  | 8  | 12  | 20  |
| Dimension of hidden layer  | 128  | 128  | 32  | 32  |

*$M$: the number of terms of the Taylor polynomial introduced in the construction of the neural networks

## Results

Pendulum Problem:
![](https://github.com/ytong6/Taylor-net/blob/master/Figures/Figure_Pend.png)

Lotka-Volterra Problem:
![](https://github.com/ytong6/Taylor-net/blob/master/Figures/LV.png)

Hénon-Heiles Problem:
![](https://github.com/ytong6/Taylor-net/blob/master/Figures/HH.png)

### Table of Losses
| Problems      | Pendulum      | Lotka-Volterra| Hénon-Heiles | Kepler |
| ------------- | ------------- | ------------- |------------- |------------- |
| Training loss | $2.75$ $\times$ $10^{-5}$  | $2.37$ $\times$ $10^{-5}$  | $9.24$ $\times$ $10^{-6}$  | $7.29$ $\times$ $10^{-5}$ |
| Testing loss | $1.39$ $\times$ $10^{-4}$  | $6.73$ $\times$ $10^{-5}$  | $9.44$ $\times$ $10^{-6}$  | $6.41$ $\times$ $10^{-5}$  |

## Dependencies
* PyTorch
* NumPy
* h5py
* Matplotlib