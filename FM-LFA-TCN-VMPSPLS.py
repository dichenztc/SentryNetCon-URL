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


def vmpspls(loss, group_member_ship, lam, gamma, alpha, t, k=100):
    groups_labels = np.array(list(set(group_member_ship)))
    b = len(groups_labels)
    selected_idx = []
    selected_score = [0] * len(loss)
    delta_t = alpha * (1 / (1 + np.exp(-t / k)))

    for j in range(b):
        idx_in_group = np.where(group_member_ship == groups_labels[j])[0]
        idx_loss_dict = {i: loss[int(i)] for i in idx_in_group}
        sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s])

        for (i, ii) in enumerate(sorted_idx_in_group):
            dynamic_threshold = lam + (gamma * delta_t) / (np.sqrt(i + 1) + np.sqrt(i))
            if loss[ii] < dynamic_threshold:
                selected_idx.append(ii)
            selected_score[ii] = loss[ii] - dynamic_threshold

    selected_idx_and_score = {idx: selected_score[idx] for idx in selected_idx}
    sorted_idx_in_selected_samples = sorted(selected_idx_and_score.keys(), key=lambda s: selected_idx_and_score[s])

    return np.array(sorted_idx_in_selected_samples)