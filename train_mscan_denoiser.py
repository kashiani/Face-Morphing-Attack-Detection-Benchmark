import os
import time
import argparse
import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from models.MSCAN import MSCAN
from utils.dataset_utils import DataLoaderTrain, DataLoaderVal
from utils.img_utils import torchPSNR

# Parse command line arguments
parser = argparse.ArgumentParser(description='Training Arguments')
parser.add_argument('-md', '--model_dir', type=str, default='./models',
                    help='Directory for saving trained models')
parser.add_argument('-td', '--train_dir', default='./dataset/training',
                    type=str, help='Directory of training dataset')
parser.add_argument('-vd', '--val_dir', default='./dataset/test',
                    type=str, help='Directory of validation dataset')
parser.add_argument('-ps', '--patch_size', type=int, default=512,
                    help='Patch size in the training phase')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('-ep', '--num_epochs', type=int, default=80,
                    help='Number of epochs for training')
parser.add_argument('-bs', '--batch_size', type=int, default=8,
                    help='Batch size for training')
parser.add_argument('-ev', '--eval_interval', type=int, default=1,
                    help='Interval for evaluation in epochs')
parser.add_argument('-rs', '--resume', action='store_true',
                    help='Resume training from the last checkpoint')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed for reproducibility
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Load datasets
print('Loading datasets...')
train_dataset = DataLoaderTrain(args.train_dir, {'patch_size': args.patch_size})
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
val_dataset = DataLoaderVal(args.val_dir, {'patch_size': args.patch_size})
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

# Determine evaluation frequency
eval_now = len(train_loader) // 8

# Initialize model and optimizer
if not args.resume:
    print('Defining model for training from scratch...')
    model = MSCAN().to(device)
    summary(model, (3, args.patch_size, args.patch_size))  # Print model summary
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    best_psnr = 0
    best_epoch = 0
    best_iter = 0
else:
    print('Resuming training from the last checkpoint...')
    model = MSCAN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint = torch.load(os.path.join(args.model_dir, "model_MSCAN_512_refined.pth"))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    start_epoch = checkpoint['epoch'] + 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.5)
    for i in range(1, checkpoint['epoch']):
        scheduler.step()
    best_psnr = checkpoint['Best PSNR']
    best_epoch = checkpoint['epoch']
    best_iter = 0

# Loss function
loss_fun = nn.MSELoss(reduction='sum')

# Training loop
print('Starting training...')
for epoch in range(start_epoch, args.num_epochs + 1):
    loss_idx = 0
    train_losses = 0
    model.train()
    scheduler.step()

    for i, data in enumerate(tqdm.tqdm(train_loader), 0):
        target = data[0].to(device)
        input = data[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        train_loss = loss_fun(output, target)
        train_losses += train_loss.item()
        loss_idx += 1
        train_loss.backward()
        optimizer.step()

        print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(epoch, loss_idx, train_loss.item()))

        # Evaluation phase
        if i % eval_now == 0 and epoch % args.eval_interval == 0:
            print('Starting evaluation...')
            model.eval()
            psnr_i = []
            with torch.no_grad():
                for ii, data_val in enumerate(val_loader, 0):
                    target = data_val[0].to(device)
                    input = data_val[1].to(device)
                    output = model(input)
                    for res, tar in zip(output, target):
                        psnr_i.append(torchPSNR(res, tar))
            psnr_i = torch.stack(psnr_i).mean().item()

            if psnr_i > best_psnr:
                best_psnr = psnr_i
                best_epoch = epoch
                best_iter = i
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'Best PSNR': best_psnr}, os.path.join(args.model_dir, "model_MSCAN_512_refined_lr0.0001.pth"))

            print("Epoch: {}, Iteration: {}, PSNR: {:.4f}, Best PSNR: {:.4f} (Epoch {}, Iteration {})".format(epoch, i, psnr_i, best_psnr, best_epoch, best_iter))
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'Best PSNR': psnr_i}, os.path.join(args.model_dir, f"model_MSCAN_512_refined_lr0.0001__epoch{epoch}.pth"))
            model.train()  # End of evaluation phase

# Compute average training loss
train_losses /= loss_idx
print("Epoch: {}, Average Training Loss: {:.4f}".format(epoch, train_losses))
torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.model_dir, "model_MSCAN_512_refined_lr0.0001.pth"))
