# Importing Libraries
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
import time

# Custom Libraries
import utils

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')


# Assuming utils, weight_init, initialize_mask, trim_weights_by_threshold,
# restore_initial_weights, evaluate_network, fit_network, and weight_masks
# are defined elsewhere.

def execute_experiment(config, iteration=0):
    # Create a unique experiment folder using timestamp or iteration
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = f"{os.getcwd()}/experiments/{config.arch_type}/{config.dataset}/exp_{timestamp}_{iteration}"
    os.makedirs(exp_dir, exist_ok=True)

    # Save config to the experiment folder
    with open(f"{exp_dir}/config.txt", 'w') as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")

    plot_dir = os.path.join(exp_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    hardware = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_weights = config.prune_type == "reinit"

    # Data Preparation

    dataset_map = {
        "mnist": (datasets.MNIST, "archs.mnist"),
        "cifar10": (datasets.CIFAR10, "archs.cifar10"),
        "fashionmnist": (datasets.FashionMNIST, "archs.mnist"),
        "cifar100": (datasets.CIFAR100, "archs.cifar100")
    }

    if config.dataset not in dataset_map:
        print(f"\nInvalid dataset: {config.dataset}\n")
        sys.exit(1)

    if config.dataset == "mnist" or config.dataset == "fashionmnist":
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet
    elif config.dataset == "cifar10":
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet
    elif config.dataset == "cifar100":
        from archs.cifar100 import AlexNet, LeNet5, fc1, vgg, resnet

    train_transform, test_transform = get_transforms(config.dataset)

    dataset_class, arch_module = dataset_map[config.dataset]
    train_data = dataset_class('../data', train=True, download=True, transform=train_transform)
    test_data = dataset_class('../data', train=False, transform=test_transform)

    train_iterator = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_iterator = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)
    bn_calibration_iterator = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)


    # Model Selection
    global network
    model_map = {
        "fc1": fc1.fc1,
        "lenet5": LeNet5.LeNet5,
        "alexnet": AlexNet.AlexNet,
        "vgg16": vgg.vgg16,
        "resnet18": resnet.resnet18,
        "densenet121": densenet.densenet121 if config.dataset == "cifar10" else None
    }

    if config.arch_type not in model_map or (config.arch_type == "densenet121" and config.dataset != "cifar10"):
        print(f"\nInvalid model: {config.arch_type}\n")
        sys.exit(1)

    network = model_map[config.arch_type]().to(hardware)
    print(network)
    # Initialize Weights
    network.apply(weight_init)

    # Save Initial Weights
    original_weights = copy.deepcopy(network.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{config.arch_type}/{config.dataset}/")
    torch.save(network,
               f"{os.getcwd()}/saves/{config.arch_type}/{config.dataset}/init_weights_{config.prune_type}.pth.tar")

    # Create Weight Masks
    initialize_mask(network)

    # Setup Optimizer and Loss
    # optim = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=1e-4)
    optim = torch.optim.SGD(network.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()

    # Display Layer Information
    for layer_name, weights in network.named_parameters():
        print(f"{layer_name}: {weights.shape}")

    # Pruning Loop
    top_accuracy = 0.0
    prune_cycles = config.prune_iterations
    compression_rates = np.zeros(prune_cycles, dtype=float)
    peak_accuracies = np.zeros(prune_cycles, dtype=float)
    losses = np.zeros(config.end_iter, dtype=float)
    accuracies = np.zeros(config.end_iter, dtype=float)

    for prune_iter in range(config.start_iter, prune_cycles):
        if prune_iter > 0:
            if config.arch_type == "vgg16" or config.arch_type == "resnet18":
                global_trim_weights_by_threshold(config.prune_percent, resample=False, reset=reset_weights)
            else:
                trim_weights_by_threshold(config.prune_percent, resample=False, reset=reset_weights)
            if reset_weights:
                network.apply(weight_init)
                counter = 0
                for layer_name, weights in network.named_parameters():
                    if 'weight' in layer_name:
                        weight_device = weights.device
                        weights.data = torch.from_numpy(weights.data.cpu().numpy() * weight_masks[counter]).to(
                            weight_device)
                        counter += 1
                counter = 0
            else:
                checkpoint_path = Path(f"{exp_dir}/model_4_init.pth")
                if checkpoint_path.exists():
                    original_weights = torch.load(checkpoint_path, map_location=hardware)
                restore_initial_weights(weight_masks, original_weights)
                recalibrate_bn(network, bn_calibration_iterator, num_batches=30)
            optim = torch.optim.SGD(network.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)

        print(f"\n--- Pruning Cycle [{iteration}:{prune_iter}/{prune_cycles}]: ---")

        # Log Non-Zero Weights
        compression = utils.print_nonzeros(network)
        compression_rates[prune_iter] = compression
        progress = tqdm(range(config.end_iter))

        for epoch in progress:
            # Evaluate Model
            if epoch % config.valid_freq == 0:
                current_accuracy = evaluate_network(network, test_iterator, loss_function)
                if current_accuracy > top_accuracy:
                    top_accuracy = current_accuracy
                    torch.save(network, f"{exp_dir}/{prune_iter}_model_{config.prune_type}.pth.tar")
            if epoch == 49 or epoch == 63:
                for g in optim.param_groups:
                    g['lr'] = g['lr'] * 0.1
            if epoch == 0:
                torch.save(network.state_dict(), f"{exp_dir}/model_4_init.pth")
            # Train Model
            current_loss = fit_network(network, train_iterator, optim, loss_function)
            losses[epoch] = current_loss
            accuracies[epoch] = current_accuracy

            # Log Progress
            if epoch % config.print_freq == 0:
                progress.set_description(
                    f"Epoch: {epoch}/{config.end_iter} Loss: {current_loss:.6f} "
                    f"Accuracy: {current_accuracy:.2f}% Best: {top_accuracy:.2f}%")

        writer.add_scalar('Accuracy/test', top_accuracy, compression)
        peak_accuracies[prune_iter] = top_accuracy

        # Plot Loss and Accuracy
        plt.figure()
        plt.plot(range(1, config.end_iter + 1), 100 * (losses - losses.min()) / losses.ptp(), c="blue",
                 label="Normalized Loss")
        plt.plot(range(1, config.end_iter + 1), accuracies, c="red", label="Accuracy")
        plt.title(f"Loss vs Accuracy ({config.dataset}, {config.arch_type})")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.grid(color="white")
        plt.savefig(f"{plot_dir}/{config.prune_type}_metrics_{compression}.png", dpi=1200)
        plt.close()

        # Save Metrics as .txt
        np.savetxt(f"{exp_dir}/{config.prune_type}_losses_{compression}.txt", losses, fmt='%.6f')
        np.savetxt(f"{exp_dir}/{config.prune_type}_accuracies_{compression}.txt", accuracies, fmt='%.6f')

        # Save Masks
        with open(f"{exp_dir}/{config.prune_type}_masks_{compression}.pkl", 'wb') as f:
            pickle.dump(weight_masks, f)

        # Reset Metrics
        top_accuracy = 0
        losses = np.zeros(config.end_iter, dtype=float)
        accuracies = np.zeros(config.end_iter, dtype=float)

    # Save Final Metrics as .txt
    np.savetxt(f"{exp_dir}/{config.prune_type}_compression.txt", compression_rates, fmt='%.6f')
    np.savetxt(f"{exp_dir}/{config.prune_type}_peak_accuracies.txt", peak_accuracies, fmt='%.6f')

    # Plot Accuracy vs Compression
    plt.figure()
    plt.plot(range(prune_cycles), peak_accuracies, c="blue", label="Top Accuracy")
    plt.title(f"Accuracy vs Remaining Weights ({config.dataset}, {config.arch_type})")
    plt.xlabel("Remaining Weights (%)")
    plt.ylabel("Test Accuracy")
    plt.xticks(range(prune_cycles), compression_rates, rotation="vertical")
    plt.ylim(min(peak_accuracies), max(peak_accuracies))
    plt.legend()
    plt.grid(color="white")
    plt.savefig(f"{plot_dir}/{config.prune_type}_accuracy_vs_compression.png", dpi=1200)
    plt.close()


def get_transforms(dataset):
    if dataset in ["mnist", "fashionmnist"]:
        # MNIST and FashionMNIST: Grayscale, single-channel
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform, transform
    elif dataset in ["cifar10", "cifar100"]:
        # CIFAR-10 and CIFAR-100: RGB, three-channel
        # Training transform with augmentations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        # Test transform (no augmentations, only normalization)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        return train_transform, test_transform
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


import torch


def fit_network(network, data_iterator, optim, loss_fn, rewinding=False):
    SMALL_VALUE = 1e-6
    hardware = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.train()
    batch_loss = None
    for idx, (images, labels) in enumerate(data_iterator):
        optim.zero_grad()
        images, labels = images.to(hardware), labels.to(hardware)
        predictions = network(images)
        batch_loss = loss_fn(predictions, labels)
        batch_loss.backward()

        # Apply mask to gradients of pruned weights
        mask_index = 0
        for param_name, param in network.named_parameters():
            if 'weight' in param_name:
                param.grad.data *= torch.from_numpy(weight_masks[mask_index]).to(hardware)
                mask_index += 1

        optim.step()
    return batch_loss.item()


def evaluate_network(network, eval_iterator, loss_fn):
    hardware = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, targets in eval_iterator:
            inputs, targets = inputs.to(hardware), targets.to(hardware)
            outputs = network(inputs)
            total_loss += F.nll_loss(outputs, targets, reduction='sum').item()
            predictions = outputs.max(1, keepdim=True)[1]
            correct_predictions += predictions.eq(targets.view_as(predictions)).sum().item()
        avg_loss = total_loss / len(eval_iterator.dataset)
        accuracy = 100. * correct_predictions / len(eval_iterator.dataset)
    return accuracy


def recalibrate_bn(model, data_loader, num_batches=40):
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            _ = model(x.cuda())

def global_trim_weights_by_threshold(threshold, resample=False, reset=False, **extras):
    global counter
    global weight_masks
    global network

    all_weights = []
    for param_name, weight in network.named_parameters():
        if 'weight' in param_name:
            array = weight.data.cpu().numpy()
            non_zero_vals = array[array != 0]
            all_weights.extend(np.abs(non_zero_vals))

    global_threshold = np.percentile(np.array(all_weights), threshold)

    counter = 0
    for param_name, weight in network.named_parameters():
        if 'weight' in param_name:
            array = weight.data.cpu().numpy()
            weight_device = weight.device

            updated_mask = np.where(np.abs(array) < global_threshold, 0, weight_masks[counter])
            weight.data = torch.from_numpy(array * updated_mask).to(weight_device)
            weight_masks[counter] = updated_mask

            counter += 1

    counter = 0


# Other
def trim_weights_by_threshold(threshold, resample=False, reset=False, **extras):
    global counter
    global weight_masks
    global network

    # Compute threshold value
    counter = 0
    for param_name, weight in network.named_parameters():
        if 'weight' in param_name:
            array = weight.data.cpu().numpy()
            non_zero_vals = array[array != 0]  # Extract non-zero elements
            threshold_val = np.percentile(np.abs(non_zero_vals), threshold)

            # Generate new mask based on threshold
            weight_device = weight.device
            updated_mask = np.where(np.abs(array) < threshold_val, 0, weight_masks[counter])

            # Update weights and mask
            weight.data = torch.from_numpy(array * updated_mask).to(weight_device)
            weight_masks[counter] = updated_mask
            counter += 1
    counter = 0


def initialize_mask(network):
    global counter
    global weight_masks
    counter = 0
    # Count number of weight parameters
    weight_count = sum(1 for name, param in network.named_parameters() if 'weight' in name)
    weight_masks = [None] * weight_count
    counter = 0
    # Create mask with ones for each weight tensor
    for name, param in network.named_parameters():
        if 'weight' in name:
            array = param.data.cpu().numpy()
            weight_masks[counter] = np.ones(array.shape, dtype=np.float32)
            counter += 1
    counter = 0


def restore_initial_weights(mask_array, original_weights):
    global network
    global counter

    counter = 0
    for param_name, weight in network.named_parameters():
        if 'weight' in param_name:
            weight_device = weight.device
            weight.data = torch.from_numpy(mask_array[counter] * original_weights[param_name].cpu().numpy()).to(
                weight_device)
            counter += 1
        else:
            weight.data = original_weights[param_name]
    counter = 0


def weight_init(module):
    """
    Initialize weights for different layer types.

    Usage:
        model = Model()
        model.apply(weight_init)
    """
    conv_normal_layers = (nn.Conv1d, nn.ConvTranspose1d)
    conv_xavier_layers = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    bn_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    recurrent_layers = (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)

    if isinstance(module, conv_normal_layers):
        init.normal_(module.weight)
        if module.bias is not None:
            init.normal_(module.bias)

    elif isinstance(module, conv_xavier_layers):
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.normal_(module.bias)

    elif isinstance(module, bn_layers):
        init.normal_(module.weight, mean=1.0, std=0.02)
        init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.normal_(module.bias)

    elif isinstance(module, recurrent_layers):
        for name, param in module.named_parameters():
            if param.data.dim() >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__ == "__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=80, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="resnet18", type=str,
                        help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=35.6, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=10, type=int, help="Pruning iterations count")


    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # FIXME resample
    resample = False
    execute_experiment(args, iteration=1)
