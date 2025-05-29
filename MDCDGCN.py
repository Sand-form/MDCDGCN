import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def normalize_A(A: torch.Tensor, symmetry: bool=False) -> torch.Tensor:
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt(d + 1e-10)
    D = torch.diag_embed(d)
    L = torch.matmul(torch.matmul(D, A), D)
    return L

class MDCDGCN(nn.Module):
    def __init__(self, S, l, class_num, hidden, device):
        super(MDCDGCN, self).__init__()

        self.S = S
        self.l = l
        self.device = device
        self.hidden = int(hidden)

        self.spectral = nn.Sequential(
            nn.Conv1d(S, hidden * 2, kernel_size=1),
            nn.BatchNorm1d(hidden * 2),
            mish(),
            nn.Conv1d(hidden * 2, hidden*2, kernel_size=1),
            nn.BatchNorm1d(hidden*2),
            mish()
        )

        self.A = nn.Parameter(torch.FloatTensor(l, l))
        nn.init.xavier_normal_(self.A)

        self.conv1 = nn.ModuleList([DeformConv2d(hidden*2, hidden, kernel_size=3, padding=1, groups=2) for _ in range(3)])
        self.conv2 = nn.ModuleList([DeformConv2d(hidden*2, hidden, kernel_size=5, padding=2, groups=2) for _ in range(3)])
        self.conv3 = nn.ModuleList([DeformConv2d(hidden*2, hidden, kernel_size=7, padding=3, groups=2) for _ in range(3)])
        self.bnc = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(9)])

        self.gcn = nn.Linear(hidden*2, hidden)
        self.aff = nn.Conv1d(hidden*2, hidden, kernel_size=2, dilation=hidden, groups=hidden)
        self.bng = nn.BatchNorm1d(hidden)

        self.fpc = nn.Conv1d(hidden, hidden, kernel_size=hidden, padding=0, groups=hidden)
        self.bnfpc = nn.BatchNorm1d(hidden)
        self.fpaff = nn.Conv1d(1, 1, kernel_size=2, padding=0, dilation=1)
        self.bnfpg = nn.BatchNorm1d(1)
        self.downsample = nn.Conv1d(hidden, hidden, kernel_size=1)

        self.output = nn.Linear(hidden, class_num)

    def DFGCM(self, x_g, L):
        x_g = self.gcn(x_g)
        res_x_g = x_g

        L = L.unsqueeze(0).expand(x_g.size(0), -1, -1)
        x_g = torch.bmm(L, x_g)

        x_g = torch.cat((x_g, res_x_g), dim=2)
        x_g = self.aff(x_g)
        x_g = self.bng(x_g)
        x_g = mish()(x_g)

        return x_g

    def MDCM(self, x_c, i):
        x_c = x_c.unsqueeze(3)

        offset0 = torch.rand(x_c.size(0), 2 * 3 * 3, x_c.size(2), x_c.size(3)).to(self.device) * 2 - 1
        x_c0 = self.conv1[i](x_c, offset0)
        x_c0 = x_c0.squeeze(3)
        x_c0 = self.bnc[i * 3 + 0](x_c0)
        x_c0 = mish()(x_c0)

        offset1 = torch.rand(x_c.size(0), 2 * 5 * 5, x_c.size(2), x_c.size(3)).to(self.device) * 2 - 1
        x_c1 = self.conv2[i](x_c, offset1)
        x_c1 = x_c1.squeeze(3)
        x_c1 = self.bnc[i * 3 + 1](x_c1)
        x_c1 = mish()(x_c1)

        offset2 = torch.rand(x_c.size(0), 2 * 7 * 7, x_c.size(2), x_c.size(3)).to(self.device) * 2 - 1
        x_c2 = self.conv3[i](x_c, offset2)
        x_c2 = x_c2.squeeze(3)
        x_c2 = self.bnc[i * 3 + 2](x_c2)
        x_c2 = mish()(x_c2)

        # x_c = (x_c1 + x_c2) / 2
        x_c = (x_c0 + x_c1 + x_c2) / 3
        x_c = F.adaptive_avg_pool1d(x_c, output_size=64)
        return x_c

    def fusion(self, x_g, x_c):
        return (x_g + x_c) / 2

    def AFPM(self, x, L):
        x_g = x
        x_c = torch.transpose(x, 1, 2)
        x_c = self.fpc(x_c)
        x_c = self.bnfpc(x_c)
        x_c = torch.transpose(x_c, 1, 2)
        x_c = mish()(x_c)

        res_x_g = x_g[:, int((x_g.size(1) - 1) / 2), :].unsqueeze(1)

        L = L.unsqueeze(0).expand(x_g.size(0), -1, -1)
        L = L[:, int((x_g.size(1) - 1) / 2), :x_g.size(1)].unsqueeze(1)
        x_g = torch.bmm(L, x_g)

        x_g = torch.cat((x_g, res_x_g), dim=2)
        x_g = self.fpaff(x_g)
        x_g = self.bnfpg(x_g)
        x_g = mish()(x_g)

        x_g = F.adaptive_avg_pool1d(x_g, output_size=64)
        x = (x_c + x_g) / 2

        x = x.squeeze(1)
        x = self.output(x)

        return x

    def forward(self, x):
        x = x.to(torch.float32)
        L = normalize_A(self.A)
        x = x.transpose(1, 2)
        x = self.spectral(x)
        x = x.transpose(1, 2)
        x_g = x
        x_g = self.DFGCM(x_g, L)
        for i in range(3):
            x_c = x
            x_c = self.MDCM(x_c, i)
            x = self.fusion(x_g, x_c)

        x = self.AFPM(x, L)

        return x
