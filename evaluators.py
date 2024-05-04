import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from sklearn import metrics

class ModelEvaluator:
    """
    Class to evaluate model performance on a given dataset using a specified loss criterion.

    Attributes:
        model (torch.nn.Module): The model to evaluate.
        criterion (function): The loss function to use for evaluation.
        device (str): Device to perform computations on ('cuda' or 'cpu').
    """
    def __init__(self, model, criterion, device='cuda'):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device

    def evaluate(self, data_loader, logger, label):
        """
        Evaluate the model on the provided data loader and compute various performance metrics including AUC, EER, BPCER, and APCER.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for evaluation.
            logger (logging.Logger): Logger for recording evaluation results.

        Returns:
            dict: A dictionary containing various evaluation metrics.
        """
        self.model.eval()
        total_loss = 0
        num_correct = 0
        num_samples = 0
        y_true = []
        y_scores = []
        iters_per_epoch = len(data_loader)

        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(data_loader), total=iters_per_epoch, desc="Evaluating", leave=True)
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, _ = self.model(inputs)
                outputs = torch.flip(outputs, [1])

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                num_correct += (predictions == labels).sum().item()
                num_samples += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probabilities[:, 1].cpu().numpy())

        accuracy = num_correct / num_samples
        average_loss = total_loss / iters_per_epoch
        fprs, tprs, thresholds = metrics.roc_curve(y_true, y_scores)
        auc = metrics.auc(fprs, tprs)
        eer_index = np.nanargmin(np.abs(tprs - (1 - fprs)))
        eer = fprs[eer_index]  # Equal error rate

        # Extract performance metrics at specific false positive rates
        def find_threshold(fprs, tprs, fnrs, target_fpr):
            idx = np.nanargmin(np.abs(fprs - target_fpr))
            return {'FPR': fprs[idx], 'TPR': tprs[idx], 'FNR': fnrs[idx]}

        fnrs = 1 - tprs
        bpcers = {f'BPCER{threshold}': find_threshold(fprs, tprs, fnrs, threshold / 100) for threshold in [1, 5, 10, 30]}
        apcers = {f'APCER{threshold}': find_threshold(fprs, fnrs, tprs, threshold / 100) for threshold in [1, 5, 10, 30]}

        # Log results
        logger.info(f"ACC: {accuracy:.4f}\tAUC: {auc:.4f} | EER: {eer:.4f}")
        logger.info(f"EER: {eer:.4f}\t" + "\t".join([f"{key}: {val['FNR']:.4f}" for key, val in bpcers.items()]))
        logger.info("\t".join([f"{key}: {val['FPR']:.4f}" for key, val in apcers.items()]))


        return {
            'accuracy': accuracy, 'average_loss': average_loss, 'auc': auc, 'eer': eer,
            'BPCER': bpcers, 'APCER': apcers, 'fprs': fprs, 'tprs': tprs, 'fnrs': fnrs
        }

class PerformancePlotter:
    """
    Class to handle plotting of performance curves like ROC and DET.
    """

    @staticmethod
    def plot_roc(fpr, tpr, auc, label, filename):
        """
        Plot ROC curve for the given FPR, TPR, and AUC values.

        Args:
            fpr (np.array): False Positive Rate values.
            tpr (np.array): True Positive Rate values.
            auc (float): Area Under the Curve value.
            label (str): Label for the plot.
            filename (str): Path to save the plot.
        """
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'{label} (AUC = {auc:.4f})')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f'ROC Curve: {label}')
        plt.legend(loc='lower right')
        plt.savefig(f"{filename}_ROC.png")
        plt.close()

    @staticmethod
    def plot_det(fpr, fnr, label, filename):
        """
        Plot DET curve for the given FPR and FNR values.

        Args:
            fpr (np.array): False Positive Rate values.
            fnr (np.array): False Negative Rate values.
            label (str): Label for the plot.
            filename (str): Path to save the plot.
        """
        plt.figure()
        plt.plot(fpr, fnr, color='blue', label=label)
        plt.xlabel("False Positive Rate (in %)")
        plt.ylabel("False Negative Rate (in %)")
        plt.title(f'DET Curve: {label}')
        plt.legend(loc='upper right')
        plt.savefig(f"{filename}_DET.png")
        plt.close()


    @staticmethod
    def plot_combined_roc(metrics_dict, filename):
        """
        Plot combined ROC curves for multiple datasets.

        Args:
            metrics_dict (dict): Dictionary containing FPR, TPR, and AUC values for each dataset.
            filename (str): Path to save the combined plot.
        """
        plt.figure()
        for label, metrics in metrics_dict.items():
            auc_label = f"{label} (AUC = {metrics['auc']:.4f}, APCER10 = {metrics['APCER']['APCER10']['FPR']:.4f})"
            print(f"Plotting {auc_label}: BPCER10={metrics['BPCER']['BPCER10']['FNR']}, APCER10={metrics['APCER']['APCER10']['FPR']}")
            plt.plot(metrics['fprs'], metrics['tprs'], label=auc_label)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title('Combined ROC Curves')
        plt.legend(loc='lower right', fontsize='small')  # Adjust legend font size
        plt.savefig(f"{filename}_combined_ROC_{label}.png")  # Include label in filename
        plt.close()

    @staticmethod
    def plot_combined_det(metrics_dict, filename):
        """
        Plot combined DET curves for multiple datasets.

        Args:
            metrics_dict (dict): Dictionary containing FPR, FNR, and AUC values for each dataset.
            filename (str): Path to save the combined plot.
        """
        plt.figure()
        for label, metrics in metrics_dict.items():
            auc_label = f"{label} (AUC = {metrics['auc']:.4f}, APCER10 = {metrics['APCER']['APCER10']['FPR']:.4f})"
            print(f"Plotting {auc_label}: BPCER10={metrics['BPCER']['BPCER10']['FNR']}, APCER10={metrics['APCER']['APCER10']['FPR']}")
            plt.plot(metrics['fprs'], metrics['fnrs'], label=auc_label)


        plt.xlabel("False Positive Rate (in %)")
        plt.ylabel("False Negative Rate (in %)")
        plt.title('Combined DET Curves')
        plt.legend(loc='upper right', fontsize='small')  # Adjust legend font size
        plt.savefig(f"{filename}_combined_DET_{label}.png")  # Include label in filename
        plt.close()

