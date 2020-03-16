import math
import torch
from torch import nn
import torch.nn.functional as F



class Bilinear(nn.Module):
    def __init__(self, inp_dim, K):
        super(Bilinear, self).__init__()
        self.w = nn.Bilinear(inp_dim, inp_dim, K, bias=True)

    def forward(self, x12):
        # import pdb; pdb.set_trace()
        L = x12.shape[-1]//2
        # return torch.einsum('ij,kjl,il->ik', x12[:, :L], self.W, x12[:, L:])
        return self.w(x12[:, :L], x12[:, L:])


class Bilinear(nn.Module):
    def __init__(self, inp_dim, K):
        super(Bilinear, self).__init__()
        self.W = nn.Parameter(torch.Tensor(K, inp_dim, inp_dim))

    def forward(self, x12):
        L = x12.shape[-1]//2
        return torch.einsum('ij,kjl,il->ik', x12[:, :L], self.W, x12[:, L:])

class Inferer(nn.Module):
    def __init__(self, inp_dim, K, mode):
        super(Inferer, self).__init__()

        self.inp_dim = inp_dim
        self.K = K
        self.mode = mode

        self.g = []

        # TODO:
        # more fusion other than cat
        if mode == 'regress':
            self.g += [
                nn.Linear(2*self.inp_dim, 2*self.inp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2*self.inp_dim, self.inp_dim)
            ]
        
        elif mode == 'linear':
            self.g += [
                nn.Linear(2*self.inp_dim, 2*self.inp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2*self.inp_dim, self.K)
            ]

            self.M = nn.Linear(self.K, self.inp_dim, bias=False)

        elif mode == 'bilinear':
            self.g += [
                Bilinear(self.inp_dim, self.K),
            ]

            self.M = nn.Linear(self.K, self.inp_dim, bias=False)

        self.g = nn.Sequential(*self.g)

        # import pdb; pdb.set_trace()

    def forward(self, q, k):
        m = self.g(torch.cat([q, k], dim=-1))
        
        if self.mode == 'regress':
            return m
        else:
            return self.M(m)


class Masker(nn.Module):
    def __init__(self, inp_dim, K, mode='bilinear', nonlin='', prior='l1'):
        super(Masker, self).__init__()
        self.inp_dim = inp_dim
        self.K = K
        self.mode = mode
        self.nonlin = nonlin
        self.prior = prior

        self.g = Inferer(inp_dim, K, mode)

    def forward(self, q, k):
        m = self.g(q, k)
        n_m = self.apply_nonlin(m)
        aux_loss = self.aux_loss(n_m, m)

        mq, mk = F.normalize(n_m*q), F.normalize(n_m*k)

        m_out = torch.einsum('ij,ij->i', mq, mk)

        return m_m, m_out, aux_loss

    def apply_nonlin(self, m):

        if self.nonlin == 'softmax':
            m = F.softmax(m, dim=-1)
        elif self.nonlin == 'relu':
            m = F.relu(m,)
        elif self.nonlin == 'tanh':
            m = F.tanh(m, dim=-1)

        return m

    def aux_loss(self, n_m, m):

        if self.prior == 'l2':
            return torch.norm(n_m, dim=-1, p=2).mean()

        elif self.prior == 'l1':
            return torch.norm(n_m, dim=-1, p=1).mean()

        elif self.prior == 'kl':
            return torch.nn.functional.kl_div(m, torch.ones(m.shape).cuda()*0.05)

        return torch.Tensor(0).cuda()

        # MI objective
        # elif 'top' in self.prior:
        #     k = int(self.prior[-1])
        #     # mask = torch.topk()


        

# class MaskedDot()