import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional
import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
import os
import torch.utils.data

#dimension
d = 1


def _check_none_zero(tensor, shape, device):
    return tensor.to(device) if tensor is not None else torch.zeros(*shape).to(device)


def to_np(x):
    return x.detach().cpu().numpy()


def Taylor_act(x):
    t1, t2 = x.shape
    y = Variable(torch.ones(t1, t2), requires_grad=True)
    k = 1
    for i in range(t1):
        k = k * (i + 1)
        y[i, :] = torch.pow(x[i, :], i) * (i + 1) / k
    return y


def four_SymInt(p0, q0, t0, t1, Tp, Vq, eps=0.001):
    n_steps = np.round((torch.abs(t1 - t0) / (eps * 4)).max().item())
    h = (t1 - t0) / n_steps
    kp = p0
    kq = q0
    c = [0.5 / (2. - 2. ** (1. / 3.)),
         (0.5 - 2. ** (-2. / 3.)) / (2. - 2. ** (1. / 3.)),
         (0.5 - 2. ** (-2. / 3.)) / (2. - 2. ** (1. / 3.)),
         0.5 / (2. - 2. ** (1. / 3.))]
    d = [1. / (2. - 2. ** (1. / 3.)),
         -2. ** (1. / 3.) / (2. - 2. ** (1. / 3.)),
         1. / (2. - 2. ** (1. / 3.)), 0.]
    for i_step in range(int(n_steps)):
        for j in range(4):
            tp = kp
            tq = kq + c[j] * Tp(kp) * h
            kp = tp - d[j] * Vq(tq) * h
            kq = tq
    return kp, kq


def plot_traj(obs=None, times=None, trajs=None, training_data=None):
    plt.clf()
    if obs is not None:
        obs = to_np(obs)
        plt.scatter(obs[:, 0], obs[:, 1], c=to_np(times.reshape(-1)))

    if training_data is not None:
        training_data= to_np(training_data)
        plt.plot(training_data[:, 0], training_data[:, 1], c='r',lw=4, label='Traning period',zorder=3)

    if trajs is not None:
        trajs = to_np(trajs)

        plt.plot(trajs[:, 0], trajs[:, 1], lw=1.5,c='C0', label='True trajectory',zorder=2)

    plt.title("",fontsize=22)
    plt.legend()
    plt.draw()
    plt.pause(0.01)


class TpTrained(nn.Module):
    def __init__(self):
        super(TpTrained, self).__init__()
        self.nx = 8
        self.params_pos = nn.Parameter(torch.randn(self.nx, d, 128) * math.sqrt(2. / (d * 128.) / self.nx),
                                       requires_grad=True)
        self.params_neg = nn.Parameter(torch.randn(self.nx, d, 128) * math.sqrt(2. / (d * 128.) / self.nx),
                                       requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, d), requires_grad=True)
        self.k = []
        k = 1
        for i in range(self.nx):
            k = k * (i + 1)
            self.k.append(k)
        self.k = torch.tensor([1] + self.k[:-1]).float().unsqueeze(-1).unsqueeze(-1)
        self.pow = torch.tensor(range(1, self.nx + 1)).float().unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        y_pos = torch.matmul(x.unsqueeze(1), self.params_pos)
        y_neg = torch.matmul(x.unsqueeze(1), self.params_neg)
        z_pos = torch.pow(y_pos, self.pow) / self.k
        z_neg = torch.pow(y_neg, self.pow) / self.k
        return self.b - torch.matmul(z_neg, self.params_neg.transpose(1, 2)).sum() + torch.matmul(z_pos,
                                                                                                  self.params_pos.transpose(
                                                                                                      1, 2)).sum()


class VqTrained(nn.Module):
    def __init__(self):
        super(VqTrained, self).__init__()
        self.nx = 8
        self.params_pos = nn.Parameter(torch.randn(self.nx, d, 128) * math.sqrt(2. / (d * 128.) / self.nx),
                                       requires_grad=True)
        self.params_neg = nn.Parameter(torch.randn(self.nx, d, 128) * math.sqrt(2. / (d * 128.) / self.nx),
                                       requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, d), requires_grad=True)
        self.k = []
        k = 1
        for i in range(self.nx):
            k = k * (i + 1)
            self.k.append(k)
        self.k = torch.tensor([1] + self.k[:-1]).float().unsqueeze(-1).unsqueeze(-1)
        self.pow = torch.tensor(range(1, self.nx + 1)).float().unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        y_pos = torch.matmul(x.unsqueeze(1), self.params_pos)
        y_neg = torch.matmul(x.unsqueeze(1), self.params_neg)
        z_pos = torch.pow(y_pos, self.pow) / self.k
        z_neg = torch.pow(y_neg, self.pow) / self.k
        return self.b - torch.matmul(z_neg, self.params_neg.transpose(1, 2)).sum() + torch.matmul(z_pos,
                                                                                                  self.params_pos.transpose(
                                                                                                      1, 2)).sum()


class NeuralODE(nn.Module):
    def __init__(self, Tp, Vq, tol=1e-3, solver=four_SymInt):
        super(NeuralODE, self).__init__()
        self.Tp = Tp
        self.Vq = Vq
        self.tol = tol
        self.solver = solver

    def forward(self, p0, q0, t0, t1):
        p, q = four_SymInt(p0, q0, t0, t1, self.Tp, self.Vq, self.tol)
        return p, q


class Tp(nn.Module):
    def forward(self, p):
        return p


class Vq(nn.Module):
    def forward(self, q):
        return torch.sin(q)



def plot_TN(Tp_t, Vq_t, f_neur, truepq,data):
    q0 = torch.tensor([[1.]])
    p0 = torch.tensor([[1.]])
    t0 = torch.tensor([[0.0]])
    times = [t0]
    neurp0 = p0
    neurq0 = q0
    plot_points = 200
    DT = 4*np.pi
    neurpq = [torch.cat([p0, q0], dim=1)]

    for i in range(plot_points):
        dt = DT / (plot_points + 0.)
        t1 = torch.tensor([[dt * (i + 1)]])
        neurp1, neurq1 = f_neur(neurp0, neurq0, t0, t1)
        times.append(t1)
        neurpq.append(torch.cat([neurp1, neurq1], dim=1))
        t0 = t1
        neurp0 = neurp1
        neurq0 = neurq1
    neurpq = torch.cat(neurpq)
    times = torch.cat(times)
    plot_traj(obs=neurpq, times=times, trajs=truepq,training_data=data)


def gen_truth(Tp_t, Vq_t,DT,plot_points):
    q0 = torch.tensor([[1.]])
    p0 = torch.tensor([[1.]])
    t0 = torch.tensor([[0.0]])
    times = [t0]
    truep0 = p0
    trueq0 = q0
    truepq = [torch.cat([p0, q0], dim=1)]

    for i in range(plot_points):
        dt = DT / (plot_points + 0.)
        t1 = torch.tensor([[dt * (i + 1)]])
        truep1, trueq1 = four_SymInt(truep0, trueq0, t0, t1, Tp_t, Vq_t, eps=0.001)
        times.append(t1)
        truepq.append(torch.cat([truep1, trueq1], dim=1))
        t0 = t1
        truep0 = truep1
        trueq0 = trueq1
    truepq = torch.cat(truepq)
    return truepq


def gen_data(n_samples=15, data_type='train'):
    Tp_t = Tp()
    Vq_t = Vq()
    Tp_t.eval()
    Vq_t.eval()
    DT = 0.01
    p0s = []
    q0s = []
    p1Ts = []
    q1Ts = []
    with torch.no_grad():
        for i in range(n_samples):
            tstart = 0.
            t0 = torch.tensor([[tstart]])
            t1 = torch.tensor([[tstart + DT]])
            p0 = 4. * torch.rand(1, 1) - 2.
            q0 = 4. * torch.rand(1, 1) - 2.
            p1T, q1T = four_SymInt(p0, q0, t0, t1, Tp_t, Vq_t, 0.001)
            p0s.append(p0)
            q0s.append(q0)
            if data_type == 'train':
                p1Ts.append(p1T)
                q1Ts.append(q1T)
            else:
                p1Ts.append(p1T)
                q1Ts.append(q1T)
    p0s = torch.cat(p0s).detach().cpu().numpy()
    q0s = torch.cat(q0s).detach().cpu().numpy()
    p1Ts = torch.cat(p1Ts).detach().cpu().numpy()        
    q1Ts = torch.cat(q1Ts).detach().cpu().numpy()

    data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    hf = h5py.File(os.path.join(data_root, data_type+ ".h5"), "w")
    hf.create_dataset('p0', data=p0s)
    hf.create_dataset('q0', data=q0s)
    hf.create_dataset('p1T', data=p1Ts)
    hf.create_dataset('q1T', data=q1Ts)
    hf.close()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type):
        datafile = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_type+'.h5')
        f = h5py.File(datafile)
        self.p0 = f['p0'][:]
        self.q0 = f['q0'][:]
        self.p1T = f['p1T'][:]
        self.q1T = f['q1T'][:]
        f.close()

        self.p0, self.q0, self.p1T, self.q1T = self.p0.astype(np.float32), self.q0.astype(np.float32), self.p1T.astype(np.float32), self.q1T.astype(np.float32)

    def __getitem__(self, index):
        return self.p0[index], self.q0[index], self.p1T[index], self.q1T[index]

    def __len__(self):
        return self.p0.shape[0]


def train():
    torch.set_printoptions(precision=10)
    Tp_t = Tp()
    Vq_t = Vq()
    Tp_t.eval()
    Vq_t.eval()
    f_neur = NeuralODE(TpTrained(), VqTrained())
    truepq = gen_truth(Tp_t, Vq_t, 4*np.pi,200)

    plt.ion()
    plt.show()
    n_steps = 300
    DT = 0.01
    training_data = gen_truth(Tp_t, Vq_t, DT,1)
    optimizer = torch.optim.Adam(f_neur.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    data_loader = torch.utils.data.DataLoader(Dataset(data_type='train'), batch_size=1, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(Dataset(data_type='test'), batch_size=1, shuffle=True)
    plt.figure(figsize=(9, 6))


    for i in range(n_steps):
        train_loss = 0
        train_sample = 0
        f_neur.train()
        for batch_index, data_batch in enumerate(data_loader):
            p0, q0, p1T, q1T = data_batch
            optimizer.zero_grad()
            tstart = 0.
            t0 = torch.tensor([[tstart]])
            t1 = torch.tensor([[tstart + DT]])
            p1N, q1N = f_neur(p0, q0, t0, t1)
            loss = torch.nn.functional.l1_loss(p1N, p1T) + torch.nn.functional.l1_loss(q1N, q1T)
            loss.backward()
            train_loss += loss.detach().cpu().item()
            train_sample += 1
            optimizer.step()
        scheduler.step()

        test_loss = 0
        test_sample = 0
        f_neur.eval()
        with torch.no_grad():
            for batch_index, data_batch in enumerate(test_data_loader):
                p0, q0, p1T, q1T = data_batch
                tstart = 0.
                t0 = torch.tensor([[tstart]])
                t1 = torch.tensor([[tstart + DT]])
                p1N, q1N = f_neur(p0, q0, t0, t1)
                loss = torch.nn.functional.l1_loss(p1N, p1T) + torch.nn.functional.l1_loss(q1N, q1T)
                test_loss += loss.detach().cpu().item()
                test_sample += 1

        if i%10==0:
            print("Epoch: {0} | Train Loss: {1} |  Test Loss: {2}".
                  format(i + 1,
                         format(train_loss / train_sample, '.2e'),
                         format(test_loss / test_sample, '.2e')))
            #visualization
            with torch.no_grad():
                  plot_TN(Tp_t, Vq_t, f_neur, truepq, training_data)

#data generation
gen_data(data_type='train')
gen_data(data_type='test')

#training
train()

