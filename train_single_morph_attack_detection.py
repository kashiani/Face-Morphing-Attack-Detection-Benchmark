
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import torch
import torch.nn as nn
from torchsummary import summary
import os
from tqdm import tqdm
from utils.img_utils import show_residual, calculate_accuracy
import logging
from datetime import datetime

def setup_logging(args):
    """
    Sets up the logging configuration.

    Args:
        args (argparse.Namespace): Parsed command line arguments containing results_name for the log file.
    """
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_results_name = os.path.join('runs', f"{args.results_name}_{time_now}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_results_name),
                            logging.StreamHandler()
                        ])
    return logging
def get_transform(image_size):
    """
    Create a torchvision transform for preprocessing images.

    Args:
        image_size (int): The target size of the image (height and width).

    Returns:
        torchvision.transforms.Compose: Composed image transformations.
    """
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])


    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Compose([
    #     transforms.Resize((image_size, image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ])



def load_datasets(args):
    """
    Load training and validation datasets with sampling for imbalanced data.

    Args:
        image_size (int): Size to which all images will be resized.
        train_dir (str): Directory containing training data.
        val_dir (str): Directory containing validation data.
        batch_size (int): Number of samples in each batch.

    Returns:
        tuple: Tuple containing training and validation DataLoader objects and the evaluation step count.
        args.image_size, args.train_dir, args.val_dir, args.batch_size
    """
    transform_op = get_transform(args.image_size)

    train_data = datasets.ImageFolder(args.train_dir, transform=transform_op)
    val_data = datasets.ImageFolder(args.val_dir, transform=transform_op)

    sampler_train = ImbalancedDatasetSampler(train_data)
    sampler_val = ImbalancedDatasetSampler(val_data)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=sampler_train)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=sampler_val)

    # eval_now = len(train_loader) // 2  # Determine evaluation frequency within an epoch
    eval_now = args.eval_frequency # Determine evaluation frequency within an epoch



    return train_loader, val_loader, eval_now

# Commented out to prevent execution in this phase
# image_size, train_dir, val_dir, batch_size = 512, "./dataset/frgc_train/train", "./dataset/test_sets/VSAPP", 8
# train_loader, val_loader, eval_now = load_datasets(image_size, train_dir, val_dir, batch_size)

def define_model(args, device):
    """
    Define and initialize the morph detection model based on the provided arguments.

    Args:
        args (argparse.Namespace): The parsed command line arguments.
        device (torch.device): The device to use for model computations.

    Returns:
        torch.nn.Module: The initialized model.
    """

    # Dynamically import the model based on the provided model name
    # The input image sizes for different models are:
    # - model_base: input size 224x224
    # - model_base_512: input size 512x512
    # - model_efficientnet: designed to work with multiple sizes, typically 512x512
    # - model_inceptionresnet: designed to work with multiple sizes, typically 512x512
    # - model_pretrained_resnet: typical input size is 512x512
    # - model_pretrained_resnet_filter: typical input size is 512x512
    # - models_512_res_baaria_REAL: input size 512x512

    print('Defining the model...')

    # Dynamically import the model based on the provided model name
    ensemble = False
    # Import the appropriate model based on the input argument
    if args.model_name == 'model_base':
        from models.model_base import MorphDetection

    elif args.model_name == 'model_base_512':
        from models.model_base_512 import MorphDetection

    elif args.model_name == 'model_base_512_avgpool':
        from models.model_base_512_avgpool import MorphDetection

    elif args.model_name == 'model_efficientnet':
        from models.model_efficientnet import MorphDetection

    elif args.model_name == 'model_inceptionresnet':
        from models.model_inceptionresnet import MorphDetection

    elif args.model_name == 'model_inceptionresnet_fc':
        from models.model_inceptionresnet import MorphDetection

    elif args.model_name == 'model_pretrained_resnet':
        from models.model_pretrained_resnet import MorphDetection

    elif args.model_name == 'model_pretrained_resnet_filter':
        from models.model_pretrained_resnet_filter import MorphDetection

    elif args.model_name == 'model_vit_B16':
        from models.model_vit_B16 import MorphDetection


    elif args.model_name == 'model_vit_L32':
        from models.model_vit_L32 import MorphDetection

    elif args.model_name == 'model_resnet_vanilla':
        from models.model_vit_L32 import MorphDetection


    elif args.model_name == 'ensemble_3models_fc':
        from models.ensemble_3models_fc import MorphDetection
        ensemble = True

    elif args.model_name == 'ensemble_3models_score':
        from models.ensemble_3models_score import MorphDetection
        ensemble = True

    else:
        raise ValueError(f"Invalid model name provided: {args.model_name}")

    model = MorphDetection(args).to(device)

    if not ensemble:
        summary(model, (3, args.image_size, args.image_size))  # Print model summary

    for index, param in enumerate(model.parameters()):
        param.requires_grad = index >= args.number_layers_freeze

    # Print the total number of layers processed
    print(f"Total layers tuned: {index + 1}")

    return model



def initialize_optimizer_and_scheduler(model, args):
    """
    Initialize the optimizer and scheduler based on command line arguments.

    Args:
        model (torch.nn.Module): The model for which the optimizer and scheduler are to be initialized.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        tuple: Returns the optimizer and scheduler initialized based on user specifications.
    """
    # Collect parameters that require gradient update
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
        else:
            print("\t no requires_grad", name)

    # Initialize the optimizer with parameters that require updates
    optimizer = torch.optim.Adam(params_to_update, lr=args.learning_rate)

    # Initialize the scheduler based on user input
    if args.scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    elif args.scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    elif args.scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max)
    elif args.scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)
    elif args.scheduler_type == 'CyclicLR':
        # Placeholder values for lower and upper lr bounds; adjust as necessary
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=args.learning_rate, step_size_up=args.scheduler_step_size)
    else:
        raise ValueError(f"Unsupported scheduler type: {args.scheduler_type}")

    return optimizer, scheduler


def train_model(model, train_loader, val_loader, eval_now, args, device, optimizer, scheduler, logging):
    """
    Train the morph detection model and evaluate its performance periodically.

    Args:
        model (torch.nn.Module): The morph detection model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        eval_now (int): Interval to perform evaluation within an epoch.
        args (argparse.Namespace): Parsed command line arguments.
        device (torch.device): The device to train the model on.

    Returns:
        None
    """
    print('Starting training...')
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, residuals = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % eval_now == 0 and i > 0:
                print('Starting evaluation...')
                evaluate_model(model, val_loader, optimizer, device, args, epoch, i, best_accuracy, logging)

        scheduler.step()
        print(f"Epoch {epoch + 1}: Average Loss: {total_loss / len(train_loader)}")

def evaluate_model(model, val_loader, optimizer, device, args, epoch, iteration, best_accuracy, logging):
    """
    Evaluate the model on the validation set and save the model if it achieves a new best accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): The device the model is running on.
        args (argparse.Namespace): Parsed command line arguments.
        epoch (int): Current epoch number.
        iteration (int): Current iteration number within the current epoch.
        best_accuracy (float): Best validation accuracy achieved so far.

    Returns:
        None
    """
    model.eval()
    with torch.no_grad():

        accuracy_i = []
        logging.info(' The Beginning of Evaluation Phase on ' + str(args.val_dir))
        for _, (inputs_val, targets_val) in enumerate(val_loader):
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            outputs_val, _ = model(inputs_val)
            accuracy_i.append(calculate_accuracy(outputs_val, targets_val))
        current_accuracy = torch.stack(accuracy_i).mean().item()

        logging.info("ACC:" + str(current_accuracy) + '\t' + "AUC {:.4f} ".format(0))

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            save_path = os.path.join(args.model_dir, f"{args.model_name}_best_epoch{epoch}_iter{iteration}.pth")

            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'accuracy': best_accuracy
            }, save_path)
            print(f"New best accuracy: {best_accuracy:.4f} saved to {save_path}")

        print("Epoch: %d Loss: %.4f" % (epoch, current_accuracy))
    model.train()


def parse_arguments():
    """
    Parse command line arguments for the morph detection training script.

    Returns:
        argparse.Namespace: The namespace containing all the arguments.
    """
    parser = argparse.ArgumentParser(description='Morph Detection Training Configuration')
    parser.add_argument('-model_dir', type=str, default='./checkpoint',
                        help='Directory to save the trained morph detection model')
    parser.add_argument('-mscan_model', type=str, default='./pretrained_weights/model_MSCAN_512_refined_lr0.0001.pth',
                        help='File path for the MSCAN model')
    parser.add_argument('-train_dir', type=str, default='./dataset/frgc_train/train',
                        help='Path to the training dataset')
    parser.add_argument('-val_dir', type=str, default='./dataset/test_sets/VSAPP',
                        help='Path to the validation dataset')
    parser.add_argument('-epochs', type=int, default=80,
                        help='Number of epochs for training')
    parser.add_argument('-batch_size', type=int, default=6,
                        help='Batch size for training')
    parser.add_argument('-learning_rate', type=float, default=5e-5,
                        help='Learning rate for optimizer')
    parser.add_argument('-scheduler_type', type=str, default='StepLR',
                        help='Type of learning rate scheduler: StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR')
    parser.add_argument('-scheduler_gamma', type=float, default=0.1,
                        help='Gamma parameter for the learning rate scheduler')
    parser.add_argument('-scheduler_step_size', type=int, default=15,
                        help='Step size for StepLR')
    parser.add_argument('-scheduler_t_max', type=int, default=50,
                        help='T_max parameter for CosineAnnealingLR')
    parser.add_argument('-scheduler_factor', type=float, default=0.1,
                        help='Factor by which the learning rate will be reduced. ReduceLROnPlateau')
    parser.add_argument('-scheduler_patience', type=int, default=10,
                        help='Number of epochs with no improvement after which learning rate will be reduced. ReduceLROnPlateau')
    parser.add_argument('-eval_frequency', type=int, default=3,
                        help='Frequency of evaluation per number of epochs')
    parser.add_argument('-show_residual', action='store_true',
                        help='Flag to show residual output')
    parser.add_argument('-image_size', type=int, default=512,
                        help='Image size (width and height) for training')
    parser.add_argument('-cuda_device', type=int, default=0,
                        help='CUDA device index')
    parser.add_argument('-mn', '--model_name', type=str, default='model_vit_B16',
                        help='Name of the model script (e.g., "model_resnet_vanilla" or "model_vit_B16" or "model_vit_L32" or\
                        "model_base" or "model_base_512" or "model_efficientnet" or \
                             "model_base_512_avgpool" or "model_pretrained_resnet" or \
                                 "model_inceptionresnet"  or "ensemble_3models_fc" or \
                                    "ensemble_3models_score"  "model_pretrained_resnet_filter"')
    parser.add_argument('-number_layers_freeze', type=int, default=0,
                        help='Number of initial layers to freeze during training')
    parser.add_argument('-results_name', type=str, default='train_results', help='Base name for results and logs')


    return parser.parse_args()



def main():
    args = parse_arguments()
    logging = setup_logging(args)
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, eval_now = load_datasets(args)
    model = define_model(args, device)
    optimizer, scheduler = initialize_optimizer_and_scheduler(model, args)
    train_model(model, train_loader, val_loader, eval_now, args, device, optimizer, scheduler, logging)


if __name__ == "__main__":
    main()

