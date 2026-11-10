
import torch
import torch.nn as nn
import torch.nn.functional as F


class TAM(nn.Module):
    

    def __init__(self,
                 in_channels: int,
                 num_segments: int,
                 alpha: int = 2,
                 adaptive_kernel_size: int = 3,
                 beta: int = 4,
                 conv1d_kernel_size: int = 3,
                 adaptive_convolution_stride: int = 1,
                 adaptive_convolution_padding: int = 1,
                 init_std: float = 0.001) -> None:
        super().__init__()

        assert beta > 0 and alpha > 0
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.alpha = alpha
        self.adaptive_kernel_size = adaptive_kernel_size
        self.beta = beta
        self.conv1d_kernel_size = conv1d_kernel_size
        self.adaptive_convolution_stride = adaptive_convolution_stride
        self.adaptive_convolution_padding = adaptive_convolution_padding
        self.init_std = init_std

        self.G = nn.Sequential(
            nn.Linear(num_segments, num_segments * alpha, bias=False),
            nn.BatchNorm1d(num_segments * alpha), nn.ReLU(inplace=True),
            nn.Linear(num_segments * alpha, adaptive_kernel_size, bias=False),
            nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels // beta,
                conv1d_kernel_size,
                stride=1,
                padding=conv1d_kernel_size // 2,
                bias=False), nn.BatchNorm1d(in_channels // beta),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // beta, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        

        n, c, h, w = x.size()
        num_segments = self.num_segments
        num_batches = n // num_segments
        assert c == self.in_channels


        x = x.view(num_batches, num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()


        theta_out = F.adaptive_avg_pool2d(
            x.view(-1, num_segments, h, w), (1, 1))


        conv_kernel = self.G(theta_out.view(-1, num_segments)).view(
            num_batches * c, 1, -1, 1)


        local_activation = self.L(theta_out.view(-1, c, num_segments)).view(
            num_batches, c, num_segments, 1, 1)


        new_x = x * local_activation


        y = F.conv2d(
            new_x.view(1, num_batches * c, num_segments, h * w),
            conv_kernel,
            bias=None,
            stride=(self.adaptive_convolution_stride, 1),
            padding=(self.adaptive_convolution_padding, 0),
            groups=num_batches * c)


        y = y.view(num_batches, c, num_segments, h, w)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)

        return y
