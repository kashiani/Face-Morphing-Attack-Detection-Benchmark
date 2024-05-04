import torch
from collections import OrderedDict


def convert_state_dict(original_state_dict):
    """
    Converts a state dictionary saved from a multi-GPU model to be compatible with a single GPU model.

    Args:
        original_state_dict (dict): The original state dictionary with 'module.' prefixes.

    Returns:
        dict: A new state dictionary with the 'module.' prefix removed from keys.
    """
    new_state_dict = OrderedDict()
    for key, value in original_state_dict.items():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_convert_checkpoint(file_path):
    """
    Load a checkpoint, rename 'model' to 'state_dict', convert it for single-GPU, and save the updated checkpoint.

    Args:
        file_path (str): Path to the original checkpoint file.

    Returns:
        None: Saves the converted checkpoint to a new file.
    """
    # Load the original checkpoint
    checkpoint = torch.load(file_path)

    # Rename 'model' to 'state_dict' and convert it
    if 'model' in checkpoint:
        state_dict = convert_state_dict(checkpoint['model'])
    else:
        raise KeyError("Checkpoint does not contain 'model' key")

    # Save the converted state dict
    new_checkpoint_path = file_path.replace('.pth', '_converted.pth')
    torch.save({'state_dict':state_dict, 'optimizer': checkpoint['optimizer']}, new_checkpoint_path)
    # torch.save({'state_dict':state_dict}, new_checkpoint_path)

    print(f"Converted checkpoint saved to {new_checkpoint_path}")


# torch.save({
#     'epoch': epoch,
#     'iteration': iteration,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict(),
#     'accuracy': best_accuracy
# }, save_path)
# Example usage
load_and_convert_checkpoint('/media/baaria/SSD8/datasets/other_system/codes/Face Morphing Attack Detection/pretrained_weights/baaria.pth')
