import torch
import torch.nn.functional as F

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

class FConv_3k7(torch.nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64), dropout_ratio=0.5):
        super(FConv_3k7, self).__init__()
        self.number_of_channels = number_of_channels
        self.conv1 = torch.nn.Conv2d(1, self.number_of_channels[0], kernel_size=7, padding=3)
        self.conv2 = torch.nn.Conv2d(self.number_of_channels[0], self.number_of_channels[1], kernel_size=7, padding=3)
        self.conv3 = torch.nn.Conv2d(self.number_of_channels[1], self.number_of_channels[2], kernel_size=7, padding=3)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_transp1 = torch.nn.ConvTranspose2d(self.number_of_channels[2], self.number_of_channels[1],
                                                     kernel_size=7, stride=2, padding=3,
                                                     output_padding=1)
        self.conv_transp2 = torch.nn.ConvTranspose2d(self.number_of_channels[1], self.number_of_channels[0],
                                                     kernel_size=7, stride=2, padding=3,
                                                     output_padding=1)
        self.conv_transp3 = torch.nn.ConvTranspose2d(self.number_of_channels[0], 1,
                                                     kernel_size=7, stride=2, padding=3,
                                                     output_padding=1)

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

class FConv_3k3(torch.nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64), dropout_ratio=0.5):
        super(FConv_3k3, self).__init__()
        self.number_of_channels = number_of_channels
        self.conv1 = torch.nn.Conv2d(1, self.number_of_channels[0], kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(self.number_of_channels[0], self.number_of_channels[1], kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(self.number_of_channels[1], self.number_of_channels[2], kernel_size=3, padding=1)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_transp1 = torch.nn.ConvTranspose2d(self.number_of_channels[2], self.number_of_channels[1],
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1)
        self.conv_transp2 = torch.nn.ConvTranspose2d(self.number_of_channels[1], self.number_of_channels[0],
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1)
        self.conv_transp3 = torch.nn.ConvTranspose2d(self.number_of_channels[0], 1,
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1)

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

class FConv_3k357(torch.nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64), dropout_ratio=0.5):
        super(FConv_3k357, self).__init__()
        self.number_of_channels = number_of_channels
        self.conv1 = torch.nn.Conv2d(1, self.number_of_channels[0], kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(self.number_of_channels[0], self.number_of_channels[1], kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(self.number_of_channels[1], self.number_of_channels[2], kernel_size=7, padding=3)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_transp1 = torch.nn.ConvTranspose2d(self.number_of_channels[2], self.number_of_channels[1],
                                                     kernel_size=7, stride=2, padding=3,
                                                     output_padding=1)
        self.conv_transp2 = torch.nn.ConvTranspose2d(self.number_of_channels[1], self.number_of_channels[0],
                                                     kernel_size=5, stride=2, padding=2,
                                                     output_padding=1)
        self.conv_transp3 = torch.nn.ConvTranspose2d(self.number_of_channels[0], 1,
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1)

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