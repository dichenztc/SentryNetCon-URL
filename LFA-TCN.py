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