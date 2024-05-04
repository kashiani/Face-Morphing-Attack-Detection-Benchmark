# Import required libraries

import os
import torch
import torch.nn as nn
import os.path
from facenet_pytorch import MTCNN, InceptionResnetV1


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
    MorphDetection model for detecting morphed images using MSCAN and InceptionResnetV1.

    This class integrates the MSCAN model for feature extraction and uses InceptionResnetV1 for final classification.

    Attributes:
        mscan_model (MSCAN): MSCAN model to preprocess the images.
        InceptionResnetV1 (InceptionResnetV1): Pretrained InceptionResnetV1 model used for classification.
    """

    def __init__(self, args, num_classes=2):
        """
        Initializes the MorphDetection model with the option to load pretrained weights for MSCAN and specifies
        the InceptionResnetV1 version and number of classes for the output layer.

        Args:
            mscan_pretrained_path (str, optional): Path to the pretrained MSCAN model weights.
            num_classes (int): Number of classes for the InceptionResnetV1 classifier.
        """

        super(MorphDetection, self).__init__()
        mscan_pretrained_path = True

        self.mscan_model = MSCAN()
        if mscan_pretrained_path:
            checkpoint = torch.load(os.path.join(args.mscan_model))
            self.mscan_model.load_state_dict(checkpoint['state_dict'])

        # Freeze all parameters in MSCAN model
        for param in self.mscan_model.parameters():
            param.requires_grad = False


        # Initialize InceptionResnetV1 with the specified version and number of classes
        self.resnet = InceptionResnetV1(pretrained='vggface2')

        self.fc = nn.Linear(512, 2, bias=True)

    def forward(self, x):
        """
        Forward pass of the MorphDetection model, which processes the input using MSCAN, computes residuals, and then
        applies InceptionResnetV1 followed by a fully connected layer for classification.

        Args:
            x (Tensor): Input tensor of images (batch size, channels, height, width).

        Returns:
            tuple: A tuple containing the classification results and the residual images.
                   - First element (Tensor): Output from the final classifier layer.
                   - Second element (Tensor): Residual images obtained by subtracting MSCAN output from input.
        """
        # Process the input image through MSCAN to obtain features
        mscan_output = self.mscan_model(x)  # batch size x 3 x 512 x 512

        # Calculate the residual by subtracting MSCAN output from the original input
        residual = x - mscan_output  # batch size x 3 x 512 x 512

        # Process the residual through InceptionResnetV1 for further feature extraction
        res_out_features = self.resnet(residual)

        # Pass the features through a fully connected layer to obtain final classification results
        classification_results = self.fc(res_out_features)

        return classification_results, residual
