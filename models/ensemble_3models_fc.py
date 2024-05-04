
from torch import nn
import torch
import pytorch_pretrained_vit
from models.model_inceptionresnet import MorphDetection as MorphDetection_base_512

class MorphDetection(nn.Module):
    """
    Integrates Vision Transformer models (ViT L_32, ViT B_16) and a custom ResNet model for feature extraction,
    freezing their parameters, and classifies images using a custom fully connected classifier.

    Attributes:
        vit_l32 (nn.Module): ViT model with L_32 configuration.
        vit_b16 (nn.Module): ViT model with B_16 configuration.
        resnet_512 (nn.Module): Custom ResNet model.
        classifier (nn.Sequential): Neural network for classification combining features from the ViT and ResNet models.
    """

    def __init__(self, args, num_classes=2):
        super().__init__()
        device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')

        # Setup ViT L_32
        self.vit_l32 = self.setup_vit('L_32_imagenet1k', device, './pretrained_weights/L_32.pth', num_classes)

        # Setup ViT B_16
        self.vit_b16 = self.setup_vit('B_16_imagenet1k', device, './pretrained_weights/B_16.pth', num_classes)

        # Setup custom ResNet 512
        self.resnet_512 = MorphDetection_base_512(args)
        self.resnet_512 = self.setup_resnet(self.resnet_512, device, './pretrained_weights/R-Hossein_model(4d255_6000hoss_6000vit)_1e-5.pth')
        # self.resnet_512 = self.setup_resnet(self.resnet_512, device, './pretrained_weights//twins_lr5e-5_MSCAN512refined_ResNetB_TestALL_epoch_ALLTESTS42.pth')


        # Classifier to combine features from all models and classify
        self.classifier = nn.Sequential(
            nn.Linear(2304, 256),  # Assuming combined features of 2304 from the context
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        ).to(device)
        self.classifier = nn.DataParallel(self.classifier)

    def setup_vit(self, model_name, device, weights_path, num_classes):
        vit_model = pytorch_pretrained_vit.ViT(model_name, pretrained=False, image_size=512, num_classes=num_classes).to(device)
        vit_model = torch.nn.DataParallel(vit_model)
        checkpoint = torch.load(weights_path)
        vit_model.load_state_dict(checkpoint["model"])
        vit_model.module.fc = torch.nn.Identity()
        return vit_model

    def setup_resnet(self, model, device, weights_path):
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.module.resnet.last_bn = nn.Identity()
        model.module.resnet.logits = nn.Identity()
        return model

    def forward(self, x512):
        y_ViT_L32 = self.vit_l32(x512)
        y_ViT_B16 = self.vit_b16(x512)
        y_base, _ = self.resnet_512(x512)

        # Apply softmax to each model's output before concatenation
        y_soft_ViT_B16 = torch.nn.functional.softmax(y_ViT_B16, dim=1)
        y_soft_ViT_L32 = torch.nn.functional.softmax(y_ViT_L32, dim=1)
        y_soft_y_base = torch.nn.functional.softmax(y_base, dim=1)

        # Concatenate outputs and classify
        y_cat_fc = torch.cat((y_soft_ViT_B16, y_soft_y_base, y_soft_ViT_L32), 1)
        output_scores = self.classifier(y_cat_fc)

        return output_scores, None
