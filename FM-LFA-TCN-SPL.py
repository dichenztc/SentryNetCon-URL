class LFA_TCN(nn.Module):
    def __init__(self, num_inputs, cnn_out_channels, cnn_kernel_size, num_channels,
                 kernel_size=2, dropout=0.2, max_length=39, attention=False):
        super(LFA_TCN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_inputs, cnn_out_channels, cnn_kernel_size, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU()
        )
        self.tcn = TemporalConvNet(cnn_out_channels, num_channels, kernel_size,
                                   dropout, max_length, attention)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.tcn(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return x


class FM(nn.Module):
    def __init__(self, n=None, k=None, pre_dim=6):
        super(FM, self).__init__()
        self.n = n
        self.v = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.linear = nn.Linear(n, pre_dim)

    def forward(self, X):
        out_1 = torch.matmul(X.view(-1, self.n), self.v.float()).pow(2).sum(dim=1, keepdim=True)
        out_2 = torch.matmul(X.view(-1, self.n).pow(2), self.v.pow(2).float()).sum(dim=1, keepdim=True)

        out_interaction = 0.5 * (out_1 - out_2).float()
        out_linear = self.linear(X.float()).float()
        out = (out_linear + out_interaction).float()

        return out

def spl(loss, lam):
    selected_idx = []
    for i, l in enumerate(loss):
        if l < lam:
            selected_idx.append(i)
    selected_idx_arr = np.array(selected_idx)
    return selected_idx_arr