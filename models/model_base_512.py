# Import required libraries

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import alexnet


class Compnent(nn.Module):
    """
    A convolutional block that applies a 2D convolution, an adaptive normalization, and a LeakyReLU activation.

    Attributes:
        conv_layer (nn.Conv2d): Convolutional layer.
        AdaptiveNorm (AdaptiveNorm): Adaptive normalization layer.
        activation (nn.LeakyReLU): LeakyReLU activation function.
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        """
        Initializes the ConvolutionalBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolutional layer.
            dilation (int): Dilation of the convolutional layer.
        """
        super(Compnent, self).__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                    padding=dilation, dilation=dilation, bias=False)
        self.AdaptiveNorm = AdaptiveNorm(out_channels)
        self.Lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """
        Forward pass of the ConvolutionalBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the convolution, normalization, and activation.
        """
        x = self.Conv(x)
        x = self.AdaptiveNorm(x)
        x = self.Lrelu(x)
        return x
#
#
class AdaptiveNorm(nn.Module):
    """
    Adaptive normalization layer that applies a learned scaling factor and mean adjustment over BatchNorm.

    Attributes:
        scale (nn.Parameter): Learned scaling factor.
        shift (nn.Parameter): Learned shift for mean adjustment.
        batch_norm (nn.BatchNorm2d): Batch normalization layer.
    """
    def __init__(self, num_features):
        """
        Initializes the AdaptiveNorm layer.

        Args:
            num_features (int): Number of features in the batch norm layer.
        """
        super(AdaptiveNorm, self).__init__()
        self.L = nn.Parameter(torch.Tensor([1.0]))  #scale
        self.M = nn.Parameter(torch.Tensor([0.0]))  #shift
        self.Bn = nn.BatchNorm2d(num_features)  # batch_norm

    def forward(self, x):
        """
        Forward pass of the AdaptiveNorm.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        return self.L * x + self.M * self.Bn(x)


class MSCAN(nn.Module):
    """
    Multi-Scale Context Aggregation Network (MSCAN) for processing images through multiple convolutional blocks.

    Attributes:
        network (nn.Sequential): Sequential container of convolutional blocks.
    """
    def __init__(self, channels=64, dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]):
        """
        Initializes the MSCAN with varying dilation factors for the convolutional blocks.

        Args:
            channels (int): Number of channels for the convolution operations.
            dilations (list of int): List of dilation factors for each convolutional block.
        """
        super(MSCAN, self).__init__()
        layers = [Compnent(3, channels, dilation=1)]  # initial layer with fixed dilation
        layers.extend(Compnent(channels, channels, dilation=d) for d in dilations)
        layers.append(Compnent(channels, 3, dilation=1))  # final layer to restore channel dimension

        self.Network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MSCAN.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Processed image tensor.
        """
        return self.Network(x)



class MorphDetection(nn.Module):
    """
    MorphDetection integrates MSCAN for preprocessing images and uses a modified AlexNet for feature extraction
    and classification. The model is intended for detecting morphed images by analyzing residuals between
    the input and its processed output.

    Attributes:
        mscan_model (MSCAN): MSCAN model for image preprocessing.
        feature_extraction (nn.Sequential): AlexNet layers for feature extraction.
        classifier (nn.Sequential): AlexNet layers modified for final classification.
    """

    def __init__(self, args, num_classes=2):
        """
        Initializes the MorphDetection model, loading MSCAN and modifying AlexNet for the task-specific pipeline.
        """
        super(MorphDetection, self).__init__()
        self.mscan_model = MSCAN()
        checkpoint = torch.load(os.path.join(args.mscan_model))
        self.mscan_model.load_state_dict(checkpoint['state_dict'])
        # Freeze all parameters in MSCAN model
        for param in self.mscan_model.parameters():
            param.requires_grad = False

        self.downsize = nn.Linear(57600, 9216, bias = True)

        # Load a pre-trained AlexNet and adapt it for feature extraction
        AlexNet = alexnet(pretrained=True)
        self.Feature_extraction_AlexNet = list(AlexNet.features)[:]
        self.Feature_extraction_AlexNet = nn.Sequential(*(self.Feature_extraction_AlexNet))

        # Freeze the parameters of feature extraction layers
        for param in self.Feature_extraction_AlexNet.parameters():
            param.requires_grad = False

        self.Feature_classifier_AlexNet = list(AlexNet.classifier)[0:3]
        self.Feature_classifier_AlexNet = nn.Sequential(*(self.Feature_classifier_AlexNet), nn.Linear(4096, 2))
        # self.Feature_extraction_AlexNet = nn.DataParallel(self.Feature_extraction_AlexNet)
        # self.Feature_classifier_AlexNet = nn.DataParallel(self.Feature_classifier_AlexNet)
        # self.downsize = nn.DataParallel(self.downsize)
        # self.mscan_model = nn.DataParallel(self.mscan_model)

    def forward(self, x):
        """
        Forward pass of the MorphDetection model.

        Args:
            x (Tensor): Input tensor of images (batch size, channels, height, width).

        Returns:
            tuple: Outputs the final classification results and the residual images.
                   - First element (Tensor): Output from the classifier.
                   - Second element (Tensor): Residual images computed as input minus MSCAN output.
        """
        output = self.mscan_model(x) # batchsize x 3 x 512 x 512
        residual = x - output # batchsize x 3 x 512 x 512

        res_out = self.Feature_extraction_AlexNet(residual)  # batchsize x 256 x 6 x 6 for 224 or batchsize x 15 x 15
        res_out_flat = res_out.view(res_out.size(0), 256 * 15 * 15) # batchsize x 9216 for 224 or
        res_out_flat_ds = self.downsize(res_out_flat)
        res_out_fc6 = self.Feature_classifier_AlexNet(res_out_flat_ds)  # batchsize x 4096 (output of fc6)

        return res_out_fc6, residual