import torch
import torch.nn.functional as F


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)


class FConv_3k5(torch.nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64), dropout_ratio=0.5):
        super(FConv_3k5, self).__init__()
        self.number_of_channels = number_of_channels
        self.conv1 = torch.nn.Conv2d(1, self.number_of_channels[0], kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(self.number_of_channels[0], self.number_of_channels[1], kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(self.number_of_channels[1], self.number_of_channels[2], kernel_size=5, padding=2)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_transp1 = torch.nn.ConvTranspose2d(self.number_of_channels[2], self.number_of_channels[1],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp2 = torch.nn.ConvTranspose2d(self.number_of_channels[1], self.number_of_channels[0],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp3 = torch.nn.ConvTranspose2d(self.number_of_channels[0], 1,
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        #self.apply(init_weights)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 1, 256, 256)
        activation1 = F.relu(self.maxpool2d(self.conv1(input_tsr)))  # (N, C1, 128, 128)
        activation2 = F.relu(self.maxpool2d(self.conv2(activation1)))  # (N, C2, 64, 64)
        activation2 = self.dropout2d(activation2)
        activation3 = F.relu(self.maxpool2d(self.conv3(activation2)))  # (N, C3, 32, 32)
        activation4 = F.relu(self.conv_transp1(activation3))  # (N, C2, 64, 64)
        activation5 = F.relu(self.conv_transp2(activation4))  # (N, C1, 128, 128)
        activation5 = self.dropout2d(activation5)
        activation6 = self.conv_transp3(activation5)  # (N, 1, 256, 256)
        output = torch.sigmoid(activation6)
        return output

class FConv_4k5(torch.nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64, 128), dropout_ratio=0.5):
        super(FConv_4k5, self).__init__()
        self.number_of_channels = number_of_channels
        self.conv1 = torch.nn.Conv2d(1, self.number_of_channels[0], kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(self.number_of_channels[0], self.number_of_channels[1], kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(self.number_of_channels[1], self.number_of_channels[2], kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(self.number_of_channels[2], self.number_of_channels[3], kernel_size=5, padding=2)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_transp1 = torch.nn.ConvTranspose2d(self.number_of_channels[3], self.number_of_channels[2],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp2 = torch.nn.ConvTranspose2d(self.number_of_channels[2], self.number_of_channels[1],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp3 = torch.nn.ConvTranspose2d(self.number_of_channels[1], self.number_of_channels[0],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp4 = torch.nn.ConvTranspose2d(self.number_of_channels[0], 1,
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 1, 256, 256)
        activation1 = F.relu(self.maxpool2d(self.conv1(input_tsr)))  # (N, C1, 128, 128)
        activation2 = F.relu(self.maxpool2d(self.conv2(activation1)))  # (N, C2, 64, 64)
        activation2 = self.dropout2d(activation2)
        activation3 = F.relu(self.maxpool2d(self.conv3(activation2)))  # (N, C3, 32, 32)
        activation4 = F.relu(self.maxpool2d(self.conv4(activation3)))  # (N, C4, 16, 16)
        activation5 = F.relu(self.conv_transp1(activation4))  # (N, C3, 32, 32)
        activation6 = F.relu(self.conv_transp2(activation5))  # (N, C2, 64, 64)
        activation6 = self.dropout2d(activation6)
        activation7 = F.relu(self.conv_transp3(activation6))  # (N, C1, 128, 128)
        activation8 = self.conv_transp4(activation7)  # (N, 1, 256, 256)
        output = torch.sigmoid(activation8)
        return output

class FConv_5k5(torch.nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64, 128, 256), dropout_ratio=0.5):
        super(FConv_5k5, self).__init__()
        self.number_of_channels = number_of_channels
        self.conv1 = torch.nn.Conv2d(1, self.number_of_channels[0], kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(self.number_of_channels[0], self.number_of_channels[1], kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(self.number_of_channels[1], self.number_of_channels[2], kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(self.number_of_channels[2], self.number_of_channels[3], kernel_size=5, padding=2)
        self.conv5 = torch.nn.Conv2d(self.number_of_channels[3], self.number_of_channels[4], kernel_size=5, padding=2)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_transp1 = torch.nn.ConvTranspose2d(self.number_of_channels[4], self.number_of_channels[3],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp2 = torch.nn.ConvTranspose2d(self.number_of_channels[3], self.number_of_channels[2],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp3 = torch.nn.ConvTranspose2d(self.number_of_channels[2], self.number_of_channels[1],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp4 = torch.nn.ConvTranspose2d(self.number_of_channels[1], self.number_of_channels[0],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp5 = torch.nn.ConvTranspose2d(self.number_of_channels[0], 1,
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 1, 256, 256)
        activation1 = self.maxpool2d(F.relu(self.conv1(input_tsr)))  # (N, C1, 128, 128)
        activation2 = self.maxpool2d(F.relu(self.conv2(activation1)))  # (N, C2, 64, 64)
        activation2 = self.dropout2d(activation2)
        activation3 = self.maxpool2d(F.relu(self.conv3(activation2)))  # (N, C3, 32, 32)
        activation4 = self.maxpool2d(F.relu(self.conv4(activation3)))  # (N, C4, 16, 16)
        activation5 = self.maxpool2d(F.relu(self.conv5(activation4)))  # (N, C5, 8, 8)
        activation6 = F.relu(self.conv_transp1(activation5))  # (N, C4, 16, 16)
        activation7 = F.relu(self.conv_transp2(activation6))  # (N, C3, 32, 32)
        activation7 = self.dropout2d(activation7)
        activation8 = F.relu(self.conv_transp3(activation7))  # (N, C2, 64, 64)
        activation9 = F.relu(self.conv_transp4(activation8))  # (N, C1, 128, 128)
        activation10 = self.conv_transp5(activation9)  # (N, 1, 256, 256)
        output = torch.clip(activation10, min=0, max=1.0)
        return output