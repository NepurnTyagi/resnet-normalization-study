import torch
from torch import nn

__all__ = ["BatchNormalisation", "InstanceNormalisation", "BatchInstanceNormalisation", "LayerNormalisation", "GroupNormalisation"]

epsilon=1e-6
momentum=0.1
G=16

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.initialize=False

    def forward(self, X):
        if not self.initialize:
            self.initialize=True
            self.w = nn.parameter.Parameter(torch.ones(X.shape[-3], device=X.device))
            self.b = nn.parameter.Parameter(torch.zeros(X.shape[-3], device=X.device))
        X = X*self.w.view(1, -1, 1, 1)+self.b.view(1, -1, 1, 1)
        return X

class BatchNormalisation(nn.Module):
    def __init__(self, affine=True):
        super().__init__()
        self.affine = affine
        self.linear = None
        if self.affine:
            self.linear = Linear()
        self.initialize = False

    def forward(self, X):
        if not self.initialize:
                self.initialize=True
                self.mu = nn.parameter.Buffer(torch.zeros(X.shape[-3], device=X.device))
                self.var = nn.parameter.Buffer(torch.ones(X.shape[-3], device=X.device))
        if self.training:
            mu = X.mean(dim=(0, -1, -2), keepdim=True)
            self.mu.mul_(1-momentum).add_(momentum*mu.squeeze(-1).squeeze(-1).squeeze(0))
            X = X-mu
            var = (X*X).mean(dim=(0, -1, -2), keepdim=True)
            self.var.mul_(1-momentum).add_(momentum*var.squeeze(-1).squeeze(-1).squeeze(0))
            X = X/torch.sqrt(var+epsilon)
        else:
            X = X-self.mu.view(1, -1, 1, 1)
            X = X/torch.sqrt(self.var.view(1, -1, 1, 1)+epsilon)
        if self.affine:
            X = self.linear(X)
        return X

class InstanceNormalisation(nn.Module):
    def __init__(self, affine=True):
        super().__init__()
        self.affine = affine
        self.linear = None
        if self.affine:
            self.linear=Linear()

    def forward(self, X):
        mu = X.mean(dim=(-1, -2), keepdim=True)
        X = X-mu
        var = (X*X).mean(dim=(-1, -2), keepdim=True)
        X = X/torch.sqrt(var+epsilon)
        if self.affine:
            X = self.linear(X)
        return X

class BatchInstanceNormalisation(nn.Module):
    def __init__(self, affine=True):
        super().__init__()
        self.batch_norm = BatchNormalisation(affine=False)
        self.instance_norm = InstanceNormalisation(affine=False)
        self.affine=affine
        self.linear=None
        if self.affine:
            self.linear=Linear()
        self.initialize=False

    def forward(self, X):
        if not self.initialize:
            self.initialize=True
            self.p = nn.parameter.Parameter(torch.zeros(X.shape[-3], device=X.device))
        p = nn.functional.sigmoid(self.p).view(1, -1, 1, 1)
        X = p*self.batch_norm(X)+(1-p)*self.instance_norm(X)
        if self.affine:
            X = self.linear(X)
        return X

class LayerNormalisation(nn.Module):
    def __init__(self, affine=True):
        super().__init__()
        self.affine=affine
        self.linear=None
        if self.affine:
            self.linear=Linear()

    def forward(self, X):
        mu = X.mean(dim=(-1, -2, -3), keepdim=True)
        X = X-mu
        var = (X*X).mean(dim=(-1, -2,-3), keepdim=True)
        X = X/torch.sqrt(var+epsilon)
        if self.affine:
            X = self.linear(X)
        return X

class GroupNormalisation(nn.Module):
    def __init__(self, affine=True):
        super().__init__()
        self.affine=affine
        self.linear=None
        if self.affine:
            self.linear=Linear()
    
    def forward(self, X):
        N, C, H, W = X.shape
        X = X.view(N, G, C//G, H, W)
        mu = X.mean(dim=(-1, -2, -3), keepdim=True)
        X = X-mu
        var = (X*X).mean(dim=(-1, -2, -3), keepdim=True)
        X = X/torch.sqrt(var+epsilon)
        X = X.view(N, C, H, W)
        if self.affine:
            X = self.linear(X)
        return X
