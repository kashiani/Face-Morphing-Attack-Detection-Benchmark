from torch import nn
import torch
import pytorch_pretrained_vit


class MorphDetection(nn.Module):
    """
    A MorphDetection class that utilizes the Vision Transformer (ViT) model from the
    pytorch_pretrained_vit library for image classification.

    Attributes:
        model (nn.Module): The Vision Transformer model pre-trained on ImageNet1k and adapted
                           for a specific number of classes and image size.
    """

    def __init__(self, args, num_classes=2):
        """
        Initializes the MorphDetection class with the specified CUDA device and number of output classes.

        Args:
            cuda_device (str): The specific CUDA device number as a string, defaults to '0'.
            num_classes (int): The number of classes for the classification task, defaults to 2.
        """
        super(MorphDetection, self).__init__()
        device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.model = pytorch_pretrained_vit.ViT('L_32_imagenet1k', pretrained=True,
                                                image_size=512, num_classes=num_classes).to(device)

    def forward(self, x):
        """
        Defines the forward pass of the ViT model on the input tensor.

        Args:
            x (torch.Tensor): The input tensor for which the output needs to be computed.

        Returns:
            tuple: A tuple containing the output from the ViT model and None as the second element.
        """
        return self.model(x), None


