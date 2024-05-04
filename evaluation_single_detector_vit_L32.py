
import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import logging
from datetime import datetime
from evaluators import ModelEvaluator, PerformancePlotter


def setup_logging(args):
    """Setup logging to file and console with dynamic filename based on timestamp and result name."""
    os.makedirs(args.results_path, exist_ok=True)
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_results_name = os.path.join(args.results_path, f"{args.model_name}_{args.results_name}_{time_now}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_results_name),
                            logging.StreamHandler()
                        ])
    return logging.getLogger('Model Evaluation')


def parse_arguments():
    """Parse command-line arguments, including multiple datasets."""
    parser = argparse.ArgumentParser(description='Model Evaluation Configuration')
    parser.add_argument('--model_name',  default = 'model_vit_L32', type=str, help='Name of the model module to import')
    parser.add_argument('--data_paths', type=str, default='FRLL_FaceMorpher:./dataset/FRLL/facemorpher,FRLL_OpenCV:./dataset/FRLL/opencv,FRLL_StyleGAN:./dataset/FRLL/stylegan',
                        help='Comma-separated paths to datasets with labels before each path separated by a colon')
    parser.add_argument('--results_path', default='./results', type=str, help='Directory to save logs and outputs')
    parser.add_argument('--results_name', default='evaluation', type=str, help='Base name for saving outputs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for data loading')
    parser.add_argument('--pretrained_weights', type=str, default ='./pretrained_weights/ViT_L_32.pth', help='Path to the trained model weights')
    parser.add_argument('--plot', default=True, type=bool, help='Whether to save combined ROC plots')
    parser.add_argument('--cuda_device', default=0, type=int, help='CUDA device index')
    args = parser.parse_args()

    # Parse data_paths into a dictionary
    args.data_paths = {dp.split(":")[0]: dp.split(":")[1] for dp in args.data_paths.split(",")}
    return args



def load_model(args, device):
    """Dynamically load the model based on the specified model name."""
    model_module = __import__("models." + args.model_name, fromlist=['MorphDetection'])
    MorphDetection = getattr(model_module, 'MorphDetection')
    model_object = MorphDetection(args)
    model_object = model_object.to(device)
    checkpoint = torch.load(args.pretrained_weights, map_location=device)
    model_object.model.load_state_dict(checkpoint['state_dict'])
    return model_object


def main():
    args = parse_arguments()
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args)

    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    model = load_model(args, device)
    criterion = nn.CrossEntropyLoss()

    metrics_dict = {}

    # Loop through the specified dataset paths
    for label, data_path in args.data_paths.items():
        dataset = datasets.ImageFolder(data_path, transform=trans)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
        evaluator = ModelEvaluator(model, criterion, device)
        metrics = evaluator.evaluate(data_loader, logger, label)
        metrics_dict[label] = metrics
        logger.info(
            f"{label} - Accuracy: {metrics['accuracy']:.4f}, Average Loss: {metrics['average_loss']:.4f}, "
            f"AUC: {metrics['auc']:.4f}, EER: {metrics['eer']:.4f}, "
            f"BPCER1: {metrics['BPCER']['BPCER1']['FNR']:.4f}, APCER1: {metrics['APCER']['APCER1']['FPR']:.4f}"
        )


    # Plot combined ROC and DET for all datasets
    if args.plot:
        PerformancePlotter.plot_combined_roc(metrics_dict, os.path.join(args.results_path, args.model_name + '_' + args.results_name))
        PerformancePlotter.plot_combined_det(metrics_dict, os.path.join(args.results_path, args.model_name + '_' + args.results_name))

    # Save the results for each dataset
    for label, metrics in metrics_dict.items():
        np.save(os.path.join(args.results_path, f"{args.results_name}_{label}_metrics.npy"), metrics)


if __name__ == "__main__":
    main()
