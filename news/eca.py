import torch
from torch import nn
from torch.nn.parameter import Parameter


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y1 = self.max_pool(x)
        y = torch.cat((y,y1), dim=1)
        y = self.conv1(y)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


if __name__ == '__main__':
    models = eca_layer(channel=512).cuda()
    input = torch.randn(2, 512, 64, 64).cuda()
    output = models(input)
    print('input_size:', input.size())
    print('output_size:', output.size())