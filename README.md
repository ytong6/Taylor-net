# Symplectic Taylor Neural Networks
* Paper: https://arxiv.org/abs/2005.04986
## Summary
We propose an effective and light-weighted learning algorithm, Symplectic Taylor Neural Networks (Taylor-nets), to conduct continuous, long-term predictions of a complex Hamiltonian dynamic system based on sparse, short-term observations. At the heart of our algorithm is a novel neural network architecture consisting of two sub-networks. Both are embedded with terms in the form of Taylor series expansion that are designed with a symmetric structure. The key mechanism underpinning our infrastructure is the strong expressiveness and special symmetric property of the Taylor series expansion, which can inherently accommodate the numerical fitting process of the spatial derivatives of the Hamiltonian as well as preserve its symplectic structure. We further incorporate a fourth-order symplectic integrator in conjunction with neural ODEs’ framework into our Taylor-net architecture to learn the continuous time evolution of the target systems while preserving their symplectic structures simultaneously. 

![](https://github.com/ytong6/Taylor-net/blob/master/Figures/net2.png)

![](https://github.com/ytong6/Taylor-net/blob/master/Figures/net.png)

## Prerequisites

Python should be installed in order to run the program. In a newly created virtual environmnent, run the following command:

```
pip install -r requirements.txt
```

All the required dependencies will then be installed.

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
| Predicting period | 2&pi;  | 20&pi;  | 10  | 20&pi;  |
| Sample size  | 15  | 25  | 25  | 25  |
| Epoch  | 100  | 150  | 100  | 50  |
| Learning rate  | 0.002  | 0.003  | 0.001  | 0.001  |
| step\_size  | 10  | 10  | 10  | 10  |
| &gamma;  | 0.8  | 0.8  | 0.8  | 0.8  |
| M*  | 8  | 8  | 12  | 20  |
| Dimension of hidden layer  | 128  | 128  | 32  | 32  |

*M: the number of terms of the Taylor polynomial introduced in the construction of the neural networks

## Results

Pendulum Problem:
 <img align="center" src="https://github.com/ytong6/Taylor-net/blob/master/Figures/Figure_Pend.png" alt="pendulem" width="700" >


Lotka-Volterra Problem:
 <img align="center" src="https://github.com/ytong6/Taylor-net/blob/master/Figures/Figure_LV.png" alt="Lotka-Volterra" width="700" >

Hénon-Heiles Problem:
 <img align="center" src="https://github.com/ytong6/Taylor-net/blob/master/Figures/Figure_HH.png" alt="Hénon-Heiles" width="700" >


### Table of Losses
| Problems      | Pendulum      | Lotka-Volterra| Hénon-Heiles | Kepler |
| ------------- | ------------- | ------------- |------------- |------------- |
| Training loss | 2.75e-05  | 2.37e-05  | 9.24e-06  | 7.29e-05 |
| Testing loss | 1.39e-04  | 6.73e-05  | 9.44e-06  | 6.41e-05  |

## N-body
   <img src="https://github.com/ytong6/Taylor-net/blob/master/Figures/n_body2.png" alt="3bodies" width="420" height="350"> <img src="https://github.com/ytong6/Taylor-net/blob/master/Figures/n_body1.png" alt="6bodies" width="420" height="370">


## Dependencies
* PyTorch
* NumPy
* h5py
* Matplotlib