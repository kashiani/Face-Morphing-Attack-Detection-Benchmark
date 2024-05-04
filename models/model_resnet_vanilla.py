# Import required libraries

import os
import torch
import torch.nn as nn
import os.path
from facenet_pytorch import MTCNN, InceptionResnetV1




class MorphDetection(nn.Module):
    """
    MorphDetection model for detecting morphed images using   InceptionResnetV1.

    This class uses InceptionResnetV1 for final classification.

    Attributes:
        InceptionResnetV1 (InceptionResnetV1): Pretrained InceptionResnetV1 model used for classification.
    """

    def __init__(self, args, num_classes=2):
        """
        Initializes the MorphDetection model specifies
        the InceptionResnetV1 version and number of classes for the output layer.

        Args:
            num_classes (int): Number of classes for the InceptionResnetV1 classifier.
        """

        super(MorphDetection, self).__init__()



        # Initialize InceptionResnetV1 with the specified version and number of classes
        self.model = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=2)


    def forward(self, x):
        """
        Forward pass of the MorphDetection model, which processes the input using  InceptionResnetV1  for classification.

        Args:
            x (Tensor): Input tensor of images (batch size, channels, height, width).

        Returns:
            tuple: A tuple containing the classification results
                   - First element (Tensor): Output for the final classification.
        """
        # Process the input image through resnet for the final classification.

        classification_results = self.model(x)

        return classification_results, None