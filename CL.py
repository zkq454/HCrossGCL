import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_fea, hid_fea, out_fea, drop_out=0.5):
        super(Classifier, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, out_fea))

    def forward(self, doc_fea):
        z = F.normalize(self.projector(doc_fea), dim=1)
        return z

class UCL(nn.Module):
    def __init__(self, in_fea: int, out_fea: int, temperature: float = 0.5, chunk_size: int = 256):
        super().__init__()
        self.temp = temperature
        self.chunk = chunk_size

        # projector
        self.proj = nn.Sequential(
            nn.Linear(in_fea, out_fea),
            nn.BatchNorm1d(out_fea),
            nn.ReLU(inplace=True),
            nn.Linear(out_fea, out_fea)
        )

        # projector_2
        self.proj_ext = nn.Sequential(
            nn.Linear(in_fea + 300, out_fea),
            nn.BatchNorm1d(out_fea),
            nn.ReLU(inplace=True),
            nn.Linear(out_fea, out_fea)
        )

    # ------------------------------------------------ public API
    def forward(self, xs):
        assert len(xs) == 3, "need 3 views(including word, entity and POS tag)"
        loss = 0.0
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            z1 = self._project(xs[i])
            z2 = self._project(xs[j])
            loss += self._nt_xent(z1, z2)
        return loss / len(pairs)

    # ------------------------------------------------ private
    def _project(self, x):
        m = self.proj if x.size(1) == self.proj[0].in_features else self.proj_ext
        return F.normalize(m(x), dim=1)

    def _nt_xent(self, z1, z2):
        device = z1.device
        N = z1.size(0)
        BS = min(self.chunk, N)
        losses = []

        for i in range(0, N, BS):
            sl = slice(i, i + BS)
            pos = torch.einsum('bd,bd->b', z1[sl], z2[sl]) / self.temp

            # neg-samples
            neg_z2 = torch.mm(z1[sl], z2.t()) / self.temp
            neg_z1 = torch.mm(z1[sl], z1.t()) / self.temp
            neg_z1.fill_diagonal_(-float('inf'))

            neg = torch.cat([neg_z2, neg_z1], dim=1)          # [BS, 2N]
            logsum = torch.logsumexp(neg, dim=1)
            losses.append(-pos + logsum)

        return torch.cat(losses).mean()