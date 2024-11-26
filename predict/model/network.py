import torch
import torch.nn as nn

class WaterLevelCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(WaterLevelCNN, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(63, 16, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(16)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        # 卷积层2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(32)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

        # 卷积层3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.instance_norm3 = nn.InstanceNorm2d(64)
        self.leaky_relu3 = nn.LeakyReLU(0.2)

        # 卷积层4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.instance_norm4 = nn.InstanceNorm2d(128)
        self.leaky_relu4 = nn.LeakyReLU(0.2)

        # 展平后的特征维度
        self.flatten_dim = 128 * input_height * input_width

        # 全连接层：将展平的特征映射到最终输出（单个水位值）
        self.fc = nn.Linear(self.flatten_dim, 1)

    def forward(self, x):
        # 第一层卷积
        x = self.conv1(x)
        x = self.instance_norm1(x)
        x = self.leaky_relu1(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = self.leaky_relu2(x)

        # 第三层卷积
        x = self.conv3(x)
        x = self.instance_norm3(x)
        x = self.leaky_relu3(x)

        # 第四层卷积
        x = self.conv4(x)
        x = self.instance_norm4(x)
        x = self.leaky_relu4(x)

        # 展平操作
        x = torch.flatten(x, 1)

        # 全连接层
        x = self.fc(x)

        return x
