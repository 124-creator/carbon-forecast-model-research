# -*- coding: utf-8 -*-
"""统计三个序列架构的可训练参数个数（每市场一个网络）。"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.stdout.reconfigure(encoding="utf-8")
NT, K = 15, 64


class QHead(nn.Module):
    def __init__(self, din, dh=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Linear(dh, NT))

    def forward(self, z):
        o = self.f(z); base = o[:, :1]
        return torch.cat([base, base + torch.cumsum(F.softplus(o[:, 1:]), 1)], 1)


class LSTMNet(nn.Module):
    def __init__(self, d=32):
        super().__init__(); self.rnn = nn.LSTM(1, d, batch_first=True); self.head = QHead(d + 2)


class TCNNet(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.c1 = nn.Conv1d(1, d, 3, padding=1, dilation=1)
        self.c2 = nn.Conv1d(d, d, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(d, d, 3, padding=4, dilation=4)
        self.head = QHead(d + 2)


class TRFNet(nn.Module):
    def __init__(self, d=32, nh=2):
        super().__init__()
        self.inp = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, K, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nh, 64, dropout=0.0, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 1)
        self.head = QHead(d + 2)


tot = 0
for nm, cls in (("LSTM", LSTMNet), ("TCN", TCNNet), ("TRF", TRFNet)):
    n = sum(p.numel() for p in cls().parameters() if p.requires_grad)
    tot += n
    print(f"{nm:<6}{n:>8,}")
print(f"{'一个市场三个架构合计':<6}{tot:>8,}   A5/A14 每市场 2 个（全池 10 个）")
