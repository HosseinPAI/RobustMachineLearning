# libraries
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.quantization import fuse_modules
import time
import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import warnings
import numfi
import copy
import torch.nn.utils.prune as prune
#warnings.filterwarnings("ignore")


# ===================================================================================================================
#                                   Normalized Weight and Put Them Between -1 and 1
# ===================================================================================================================
def clamp_weights(model, min_value=-1, max_value=1):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):  # Focus on conv and linear layers
            with torch.no_grad():
                module.weight.data.clamp_(min_value, max_value)


# ===================================================================================================================
#                                               Fuse Layer Model
# ===================================================================================================================
def fuse_model(model):
    # Fuse the first conv and bn layers
    fuse_modules(model, ['conv1', 'bn1'], inplace=True)
    # Fuse Conv2d + BatchNorm2d + optional ReLU within each BasicBlock of ResNet layers
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                # Corrected to fuse conv1 + bn1, and separately conv2 + bn2
                fuse_modules(basic_block, ['conv1', 'bn1'], inplace=True)
                fuse_modules(basic_block, ['conv2', 'bn2'], inplace=True)

                # Fusing shortcut layers if present and has more than 1 layer (conv + bn)
                if hasattr(basic_block.shortcut, 'named_children'):
                    for name, sub_module in basic_block.shortcut.named_children():
                        if len(list(sub_module.named_children())) > 1:  # Checking if shortcut has conv + bn
                            fuse_modules(basic_block.shortcut, [name], inplace=True)


# ===================================================================================================================
#                   Some Functions for converting from integer to 2's complement and Bit Flipping
# ===================================================================================================================
def to_twos_complement(n, bits):
    """Convert an integer to two's complement binary representation."""
    # Mask to get 'bits' bits
    mask = 2 ** bits - 1
    if n < 0:
        n = ((abs(n) ^ mask) + 1) & mask
    return n


def from_twos_complement(n, bits):
    """Convert a two's complement binary back to an integer."""
    # Check if the sign bit is set (negative number)
    if n & (1 << (bits - 1)):
        return n - (1 << bits)
    return n


def flip_sign_bit(n, bits):
    """Flip the sign bit of a binary number."""
    # Flip the MSB
    return n ^ (1 << (bits - 1))


def flip_random_bit_in_integer(value, bits):
    """ Flip a random bit in an 8-bit integer and return the new integer """
    bit_to_flip = random.randint(0, bits-1)  # Choose a random bit position from 0 to 7
    mask = 1 << bit_to_flip            # Create a mask with only that bit set
    new_value = value ^ mask           # Flip the bit using XOR
    return new_value


# ===================================================================================================================
#                                       Compare Two models after and before Changing
# ===================================================================================================================
def compare_model_weights(model_before, model_after):
    # Retrieve state dictionaries
    state_dict_before = model_before.state_dict()
    state_dict_after = model_after.state_dict()

    changes = {}
    # Compare state dictionaries
    for key in state_dict_before:
        if torch.any(state_dict_before[key] != state_dict_after[key]):
            changes[key] = (state_dict_before[key], state_dict_after[key])

    return changes


# ===================================================================================================================
#                                      Find Top K Weights and Change Them in the Model
# ===================================================================================================================
def find_and_change_important_weights(model, layer_gradients, args, device):
    names_list = []
    param_list = []

    # Collect gradients from convolutional and linear layers
    for name, param in model.named_parameters():
        if (('weight' in name) and (param.grad is not None) and ('conv' in name or 'linear' in name or 'fc' in name) and
                ('shortcut' not in name) and ('downsample' not in name)):
            names_list.append(name)
            param_list.append(param)

    all_grads = torch.cat(layer_gradients)
    _, idxs = torch.topk(all_grads, args.num_of_weights)

    important_weights = []
    current_index = 0
    for idx, (grad, name, param) in enumerate(zip(layer_gradients, names_list, param_list)):
        length = grad.numel()
        mask = (idxs >= current_index) & (idxs < current_index + length)
        #print(f'Layer name: {name} -> {torch.sum(mask).item()}')
        #print('---------------------------------------------------')
        selected_idxs = idxs[mask] - current_index
        for i in selected_idxs:
            layer_name, param_type = name.rsplit('.', 1)
            important_weights.append((param, int(i), layer_name, param_type))
        current_index += length

    for param, idx, layer_name, param_type in important_weights:
        model_module = dict(model.named_modules())[layer_name]

        if isinstance(model_module, (torch.nn.Conv2d, torch.nn.Linear)) and param_type == 'weight':
            model_weight = getattr(model_module, param_type)
            # print(f'Model weight: {model_weight.shape} \n {model_weight}')
            if len(model_weight.shape) == 4:
                channel = int(idx // (model_weight.shape[1] * model_weight.shape[2] * model_weight.shape[3]))
                temp = int(idx - channel * (model_weight.shape[1] * model_weight.shape[2] * model_weight.shape[3]))
                ic = int(temp // (model_weight.shape[2] * model_weight.shape[3]))
                temp = int(temp - (ic * (model_weight.shape[2] * model_weight.shape[3])))
                i = int(temp // model_weight.shape[3])
                j = int(temp % model_weight.shape[3])
                original_float_value = model_weight[channel][ic][i][j].clone().detach()
            else:
                channel = int(idx // model_weight.shape[1])
                i = int(idx % model_weight.shape[1])
                original_float_value = model_weight[channel][i].clone().detach()

            # print(f'Original Float Value: {original_float_value.cpu().detach().numpy()}')
            binary = numfi(original_float_value.cpu().detach().numpy(), 1, args.total, args.fraction).bin
            # print(f'Binary: {binary}')
            # Bit flipping
            binary_twos_complement = to_twos_complement(int(binary[0], 2), args.total)
            flipped_binary = flip_sign_bit(binary_twos_complement, args.total)
            converted_back = from_twos_complement(flipped_binary, args.total)
            # print(f'flipped int value: {converted_back}')
            new_float_value = (torch.tensor(converted_back) / (2 ** args.fraction)).clone().detach().to(device)
            # print(f'flipped float value: {new_float_value}')
            # update weight
            model_weight = param.data.view(-1)
            updated_weight = model_weight.clone()
            updated_weight[idx] = new_float_value
            param.data = updated_weight.view(param.data.size())
            # print('------------------------------------------------------')
    return


# ===================================================================================================================
#                                       Find Top K Weights Based on the Gradients
# ===================================================================================================================
def find_important_weight(model, layer_gradients, args, mode='important'):
    names_list = []
    param_list = []

    # Collect gradients from convolutional and linear layers
    for name, param in model.named_parameters():
        if (('weight' in name) and (param.grad is not None) and ('conv' in name or 'linear' in name or 'fc' in name) and
                ('shortcut' not in name) and ('downsample' not in name)):
            names_list.append(name)
            param_list.append(param)

    all_grads = torch.cat(layer_gradients)
    if mode == 'important':
        _, idxs = torch.topk(all_grads, args.num_of_weights)
    else:
        total_elements = all_grads.numel()
        idxs = torch.randperm(total_elements, device=all_grads.device)[:args.num_of_weights]

    important_weights = []
    current_index = 0
    for idx, (grad, name, param) in enumerate(zip(layer_gradients, names_list, param_list)):
        length = grad.numel()
        mask = (idxs >= current_index) & (idxs < current_index + length)
        selected_idxs = idxs[mask] - current_index
        for i in selected_idxs:
            layer_name, param_type = name.rsplit('.', 1)
            important_weights.append((param, int(i), layer_name, param_type))
        current_index += length

    return important_weights


# ===================================================================================================================
#                                       Test Model with Test Dataset
# ===================================================================================================================
def test_model(model, test_loader, device):
    model.eval()
    test_correct = 0
    test_total = 0
    valid_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    return test_accuracy


# ===================================================================================================================
#                          Train, Validation, and Test Process for Standard Model
# ===================================================================================================================
def train_and_validate(train_loader, valid_loader, test_loader, device, args, round_num):
    # Initialize Model
    print(f'Model Initialization in Round {round_num} ...')
    model = torchvision.models.resnet18(pretrained=False).to(device)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    model.maxpool = nn.Identity().to(device)
    model.fc = nn.Linear(model.fc.in_features, args.num_of_class).to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Placeholder for metrics
    train_ls = []
    valid_ls = []
    train_acc = []
    valid_acc = []

    print(f'Start Training in Round {round_num} ...\n')
    best_acc = 0
    epc = 0
    for ep in range(args.epochs):
        start_time = time.time()
        # ====================================================
        #                    Train Process
        # ====================================================
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            train_running_loss += train_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # ====================================================
        #                    Valid Process
        # ====================================================
        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss = criterion(outputs, labels)
                valid_running_loss += valid_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        # ====================================================
        #                    Save Results
        # ====================================================
        train_avg_loss = train_running_loss / len(train_loader)
        valid_avg_loss = valid_running_loss / len(valid_loader)
        train_accuracy = 100.0 * train_correct / train_total
        valid_accuracy = 100.0 * valid_correct / valid_total
        train_ls.append(train_avg_loss)
        valid_ls.append(valid_avg_loss)
        train_acc.append(train_accuracy)
        valid_acc.append(valid_accuracy)

        # Save Best Model
        if valid_accuracy > best_acc:
            epc = ep
            best_acc = valid_accuracy
            torch.save(model.state_dict(), args.save_path + args.standard_model_name)

        print(f"Epoch {ep + 1}/{args.epochs} in Round({round_num}): "
              f"(Time = {(time.time() - start_time) / 60.0:.2f} min)\n"
              f" Train Loss: {train_avg_loss:.6f}\t Train Accuracy: {train_accuracy:.2f}%\n"
              f" Valid Loss: {valid_avg_loss:.6f}\t Valid Accuracy: {valid_accuracy:.2f}%\n")

        scheduler.step()

    test_acc = test_model(model, test_loader, device)
    print(f'In Epoch {epc}, The Best Validation Accuracy in Round{round_num}: {best_acc: .3f}')
    print(f'The Test Accuracy in Round{round_num}: {test_acc: .3f}')

    torch.save(train_ls, args.save_path + "train_loss_standard_{0}.pt".format(round_num))
    torch.save(valid_ls, args.save_path + "valid_loss_standard_{0}.pt".format(round_num))
    torch.save(train_acc, args.save_path + "train_acc_standard_{0}.pt".format(round_num))
    torch.save(valid_acc, args.save_path + "valid_acc_standard_{0}.pt".format(round_num))
    torch.save(best_acc, args.save_path + "best_acc_standard_{0}.pt".format(round_num))
    torch.save(test_acc, args.save_path + "test_acc_standard_{0}.pt".format(round_num))

    return test_acc


# ===================================================================================================================
#                       Train, Validation, and Test Process for Adding Noise and Pruning to the System
# ===================================================================================================================
def apply_pruning(model, pruning_rate):
    print(f"Applying pruning with rate {pruning_rate:.2f}...")
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            with torch.no_grad():
                weight = module.weight.data
                threshold = torch.topk(weight.abs().view(-1), int(weight.numel() * pruning_rate),
                                       largest=False).values.max()
                mask = (weight.abs() > threshold).float()
                module.weight.data.mul_(mask)

                num_zeros = (module.weight.data == 0).sum().item()
                total = module.weight.data.numel()
                print(f"Layer {name}: {100.0 * num_zeros / total:.2f}% sparsity ({num_zeros}/{total} zero weights)")


def prune_model_by_gradients_and_weights(model, gradients, pruning_rate):
    global_importance = []
    param_to_layer_map = []

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and name in gradients:
            with torch.no_grad():
                grad = gradients[name]
                weight = module.weight.data
                importance = torch.abs(grad * weight)  # Combined metric

                global_importance.append(importance.view(-1))
                param_to_layer_map.extend([(name, i) for i in range(importance.numel())])

    global_importance = torch.cat(global_importance)

    total_weights = global_importance.numel()
    num_to_prune = int(total_weights * pruning_rate)
    if num_to_prune == 0:
        return
    global_threshold = torch.topk(global_importance, num_to_prune, largest=False).values.max()

    total_zeros = 0
    total_params = 0
    sparsity_by_layer = {}

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and name in gradients:
            with torch.no_grad():
                grad = gradients[name]
                weight = module.weight.data
                importance = torch.abs(grad * weight)

                mask = (importance > global_threshold).float()
                module.weight.data.mul_(mask)

                num_zeros = (module.weight.data == 0).sum().item()
                num_params = module.weight.data.numel()
                sparsity = 100.0 * num_zeros / num_params

                total_zeros += num_zeros
                total_params += num_params

                sparsity_by_layer[name] = sparsity
    '''
    # Report sparsity
    print("\nLayer-wise Sparsity:\n-------------------")
    for layer_name, sparsity in sparsity_by_layer.items():
        print(f"{layer_name}: {sparsity:.2f}% sparsity")

    overall_sparsity = 100.0 * total_zeros / total_params if total_params > 0 else 0.0
    print("\nOverall Model Sparsity:\n-----------------------")
    print(f"Overall Sparsity: {overall_sparsity:.2f}% ({total_zeros}/{total_params} zero weights)")
    '''


def calculate_model_sparsity(device, mode='standard'):
    # Load the model
    model = torchvision.models.resnet18(pretrained=False).to(device)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    model.maxpool = nn.Identity().to(device)
    model.fc = nn.Linear(model.fc.in_features, args.num_of_class).to(device)
    if mode == 'standard':
        model.load_state_dict(torch.load(args.save_path + args.standard_model_name, map_location=torch.device('cpu')))
        title = 'Standard Model'
    elif mode == 'pruned':
        model.load_state_dict(torch.load(args.save_path + args.pruned_model_name, map_location=torch.device('cpu')))
        title = 'Pruned Model'
    elif mode == 'fault_aware':
        model.load_state_dict(torch.load(args.save_path + args.standard_noisy_model_name, map_location=torch.device('cpu')))
        title = 'Fault-aware Training Model with Pruning'

    model.eval()
    total_params = 0
    zero_params = 0
    layer_sparsity = {}
    print(f"\nLayer-wise Sparsity for {title}:\n")
    for name, param in model.named_parameters():
        if 'weight' in name:  # Focus on weight tensors
            total = param.numel()
            zeros = (param == 0).sum().item()
            sparsity = 100.0 * zeros / total
            total_params += total
            zero_params += zeros

            print(f"{name}: {sparsity:.2f}% sparsity ({zeros}/{total} zero weights)")
            layer_sparsity[name] = sparsity

    overall_sparsity = 100.0 * zero_params / total_params if total_params > 0 else 0
    print(f"\nOverall Model Sparsity for {title}:\n")
    print(f"Overall Sparsity: {overall_sparsity:.2f}% ({zero_params}/{total_params} zero weights)")

    return {
        "layer_sparsity": layer_sparsity,
        "overall_sparsity": overall_sparsity
    }


def fault_aware_train_validation(train_loader, valid_loader, test_loader, device, args, round_num, pruning_rate=0.5,
                                 mode='standard'):
    # Initialize ResNet-20 model
    print(f'Model Initialization in Round {round_num} ...')
    model = torchvision.models.resnet18(pretrained=False).to(device)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    model.maxpool = nn.Identity().to(device)
    model.fc = nn.Linear(model.fc.in_features, args.num_of_class).to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Placeholder for metrics
    train_ls = []
    valid_ls = []
    train_acc = []
    valid_acc = []

    print(f'Start Training in Round {round_num} ...\n')
    layer_list = []
    for name, module in model.named_modules():
        if ((isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)) and
                ('shortcut' not in name) and ('downsample' not in name)):
            layer_list.append(name)

    w_cnt = 0
    best_acc = 0
    epc = 0
    print(f'The epoch number to start pruning: {args.start_pruning_epoch}')
    for ep in range(args.epochs):
        start_time = time.time()
        if ep >= args.start_pruning_epoch:
            current_pruning_rate = min(
                0.1 + ((ep - args.start_pruning_epoch) / (args.epochs - args.start_pruning_epoch)) * 0.7,
                pruning_rate)  # Start at 0.1 and increase to max 0.5
        else:
            current_pruning_rate = 0.1
        print(f'Current Pruning Rate in epoch = {ep} is {current_pruning_rate}')
        # ====================================================
        #                    Train Process
        # ====================================================
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        step = 0
        noise_cnt = 0
        cnt = 0
        initial = 1
        layer_gradients = {name: torch.zeros(module.weight.shape, device=device) for name, module in model.named_modules()
                           if ((isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)) and
                               ('downsample' not in name) and ('shortcut' not in name))}

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            train_running_loss += train_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            # =========================================================
            #   Find the Top K weights based on the maximum gradient
            # =========================================================
            # Collect gradients for convolutional layers
            for name, param in model.named_parameters():
                result = name.split('.')
                layer_name = ''
                if 'bn' not in name:
                    #print(result)
                    if len(result) == 2 and result[-1] == 'weight':
                        layer_name = result[0]
                    elif len(result) == 4 and result[-1] == 'weight':
                        layer_name = str(result[0] + '.' + result[1] + '.' + result[2])

                    if layer_name in layer_list:
                        if initial:
                            layer_gradients[layer_name] = torch.abs(param.grad.data.clone())
                        else:
                            layer_gradients[layer_name] += torch.abs(param.grad.data.clone())

            initial = 0
            gradients_for_pruning = copy.deepcopy(layer_gradients)

            if ep > args.start_noise_point:
                if (step + 1) % 2 == 0:
                    cnt += 1
                    random_number = random.randint(0, 1)
                    if random_number == 1:
                        w_cnt += 1
                        noise_cnt += 1

                        normalized_gradients = {}
                        for name, grad in layer_gradients.items():
                            norm = grad.norm()
                            normalized_gradients[name] = torch.abs_(layer_gradients[name] / (norm + 1e-10))

                        all_grad = []
                        for name in layer_list:
                            all_grad.append(normalized_gradients[name].view(-1))
                        find_and_change_important_weights(model, all_grad, args, device)
                        layer_gradients = {name: torch.zeros(module.weight.shape, device=device) for name, module in
                                           model.named_modules() if ((isinstance(module, torch.nn.Conv2d) or
                                                                      isinstance(module, torch.nn.Linear)) and
                                                                     ('downsample' not in name) and
                                                                     ('shortcut' not in name))}
                        initial = 1

            optimizer.step()

            if mode == 'BitMapped':
                clamp_weights(model)

            step += 1

        if ep >= args.start_pruning_epoch:
            prune_model_by_gradients_and_weights(model, gradients_for_pruning, current_pruning_rate)
            # apply_pruning(model, current_pruning_rate)
            # calculate_model_sparsity(device, mode='fault_aware')

        # ====================================================
        #                    Valid Process
        # ====================================================
        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss = criterion(outputs, labels)
                valid_running_loss += valid_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        # ====================================================
        #                    Save Results
        # ====================================================
        train_avg_loss = train_running_loss / len(train_loader)
        valid_avg_loss = valid_running_loss / len(valid_loader)
        train_accuracy = 100.0 * train_correct / train_total
        valid_accuracy = 100.0 * valid_correct / valid_total
        train_ls.append(train_avg_loss)
        valid_ls.append(valid_avg_loss)
        train_acc.append(train_accuracy)
        valid_acc.append(valid_accuracy)

        # Save Best Model
        if ep >= args.start_pruning_epoch:
            if valid_accuracy > best_acc:
                best_acc = valid_accuracy
                epc = ep
                torch.save(model.state_dict(), args.save_path + args.standard_noisy_model_name)

        print(f"Epoch {ep + 1}/{args.epochs} in Round({round_num}): "
              f"(Time = {(time.time() - start_time) / 60.0:.2f} min)\n"
              f" Train Loss: {train_avg_loss:.6f}\t Train Accuracy: {train_accuracy:.2f}%\n"
              f" Valid Loss: {valid_avg_loss:.6f}\t Valid Accuracy: {valid_accuracy:.2f}%\n"
              f" Number of Noise Adding Process: {noise_cnt}/{cnt}")

        scheduler.step()

    test_acc = test_model(model, test_loader, device)
    print(f'In Epoch {epc}, The Best Validation Accuracy in Round{round_num}: {best_acc: .3f}')
    print(f'The Test Accuracy in Round{round_num}: {test_acc: .3f}')
    torch.save(train_ls, args.save_path + "noisy_train_loss_{0}_{1}.pt".format(mode, round_num))
    torch.save(valid_ls, args.save_path + "noisy_valid_loss_{0}_{1}.pt".format(mode, round_num))
    torch.save(train_acc, args.save_path + "noisy_train_acc_{0}_{1}.pt".format(mode, round_num))
    torch.save(valid_acc, args.save_path + "noisy_valid_acc_{0}_{1}.pt".format(mode, round_num))
    torch.save(best_acc, args.save_path + "best_acc_noisy_{0}_{1}.pt".format(mode, round_num))
    torch.save(test_acc, args.save_path + "test_acc_noisy_{0}_{1}.pt".format(mode, round_num))

    return test_acc

# ===================================================================================================================
#                                                    Attack Model
# ===================================================================================================================
def attack(valid_loader, test_loader, device, args, round_num, mode='standard', attack_type='sign',
           attack_scenario='important'):
    model = torchvision.models.resnet18(pretrained=False).to(device)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    model.maxpool = nn.Identity().to(device)
    model.fc = nn.Linear(model.fc.in_features, args.num_of_class).to(device)

    if mode == 'standard':
        print(f'Load The The standard model without any special training or bit mapping in Round {round_num}'
              f' For {attack_type} attack')
        model.load_state_dict(torch.load(args.save_path + args.standard_model_name, map_location=torch.device('cpu')))
    elif mode == 'pruned':
        print(f'Load The The pruned model without any special training or bit mapping in Round {round_num}'
              f' For {attack_type} attack')
        model.load_state_dict(torch.load(args.save_path + args.pruned_model_name, map_location=torch.device('cpu')))
    elif mode == 'fault_aware':
        print(f'Load The The standard model with fault_aware training in Round {round_num}'
              f' For {attack_type} attack')
        model.load_state_dict(
            torch.load(args.save_path + args.standard_noisy_model_name, map_location=torch.device('cpu')))

    start_time = time.time()
    test_acc = test_model(model, test_loader, device)
    layer_gradients = {name: torch.zeros(module.weight.shape, device=device) for name, module in model.named_modules()
                       if ((isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)) and
                           ('downsample' not in name) and ('shortcut' not in name))}

    layer_list = []
    for name, module in model.named_modules():
        if ((isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)) and
                ('shortcut' not in name) and ('downsample' not in name)):
            layer_list.append(name)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    initial = 1
    for images, labels in tqdm(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        # Collect gradients for convolutional layers
        for name, param in model.named_parameters():
            result = name.split('.')
            layer_name = ''
            if 'bn' not in name:
                if len(result) == 2 and result[-1] == 'weight':
                    layer_name = result[0]
                elif len(result) == 4 and result[-1] == 'weight':
                    layer_name = str(result[0] + '.' + result[1] + '.' + result[2])

                if layer_name in layer_list:
                    if initial:
                        layer_gradients[layer_name] = torch.abs(param.grad.data.clone())
                    else:
                        layer_gradients[layer_name] += torch.abs(param.grad.data.clone())

        initial = 0

    normalized_gradients = {}
    for name, grad in layer_gradients.items():
        norm = grad.norm()
        normalized_gradients[name] = torch.abs_(layer_gradients[name] / (norm + 1e-10))

    all_grad = []
    for name in layer_list:
        all_grad.append(normalized_gradients[name].view(-1).to(device))

    important_weights = find_important_weight(model, all_grad, args, mode=attack_scenario)

    # print(important_weights)
    print(f" >> Calculate the value of loss for each important weight in mode > {mode} < in Round: {round_num}. \n")
    loss = []
    index = 0
    with torch.no_grad():
        for param, idx, layer_name, param_type in important_weights:
            model = torchvision.models.resnet18(pretrained=False).to(device)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
            model.maxpool = nn.Identity().to(device)
            model.fc = nn.Linear(model.fc.in_features, args.num_of_class).to(device)

            if mode == 'standard':
                model.load_state_dict(
                    torch.load(args.save_path + args.standard_model_name, map_location=torch.device('cpu')))
            elif mode == 'pruned':
                model.load_state_dict(
                    torch.load(args.save_path + args.pruned_model_name, map_location=torch.device('cpu')))
            elif mode == 'fault_aware':
                model.load_state_dict(
                    torch.load(args.save_path + args.standard_noisy_model_name, map_location=torch.device('cpu')))

            # model_before = copy.deepcopy(model)
            model_module = dict(model.named_modules())[layer_name]
            index += 1

            if isinstance(model_module, (torch.nn.Conv2d, torch.nn.Linear)) and param_type == 'weight':
                model_weight = getattr(model_module, param_type)

                if len(model_weight.shape) == 4:
                    channel = int(idx // (model_weight.shape[1] * model_weight.shape[2] * model_weight.shape[3]))
                    temp = int(idx - channel * (model_weight.shape[1] * model_weight.shape[2] * model_weight.shape[3]))
                    ic = int(temp // (model_weight.shape[2] * model_weight.shape[3]))
                    temp = int(temp - (ic * (model_weight.shape[2] * model_weight.shape[3])))
                    i = int(temp // model_weight.shape[3])
                    j = int(temp % model_weight.shape[3])
                    original_float_value = model_weight[channel][ic][i][j].clone().detach()
                else:
                    channel = int(idx // model_weight.shape[1])
                    i = int(idx % model_weight.shape[1])
                    original_float_value = model_weight[channel][i].clone().detach()

                # print(f'Original Float Value: {original_float_value.cpu().detach().numpy()}')
                binary = numfi(original_float_value.cpu().detach().numpy(), 1, args.total, args.fraction).bin
                # print(f'Binary: {binary}')
                if attack_type == 'sign':
                    binary_twos_complement = to_twos_complement(int(binary[0], 2), args.total)
                    flipped_binary = flip_sign_bit(binary_twos_complement, args.total)
                    converted_back = from_twos_complement(flipped_binary, args.total)
                elif attack_type == 'random':
                    binary_twos_complement = to_twos_complement(int(binary[0], 2), args.total)
                    flipped_binary = flip_random_bit_in_integer(binary_twos_complement, args.total)
                    converted_back = from_twos_complement(flipped_binary, args.total)
                else:
                    random_number = random.randint(0, 256)
                    converted_back = from_twos_complement(random_number, args.total)

                new_float_value = (torch.tensor(converted_back) / (2 ** args.fraction)).clone().detach().to(device)
                # update weight
                weights = model_module.weight.data.view(-1)
                weights[idx] = new_float_value
                model_module.weight.data = weights.view(model_module.weight.data.size())
                # print('------------------------------------------------------')
                model.eval()
                loss_temp = 0
                for data, target in tqdm(valid_loader):
                    data = data.to(device)
                    target = target.to(device)
                    target_pred = model(data)
                    loss_temp += criterion(target_pred, target)

                loss.append((loss_temp, param, idx, layer_name, param_type, new_float_value))
                print(f'For weight number {index} with Float value = {original_float_value: .5f}, '
                      f'and Binary value = {binary} and then change to {new_float_value}, '
                      f'the total loss is : {loss_temp:.4f}')

    loss_sorted = sorted(loss, key=lambda x: x[0], reverse=True)

    print("===========================================================================================")
    print("===========================================================================================")
    print(f" >> Check the accuracy of the model based on the top loss "
          f"generated in model > {mode} <in round:{round_num}. \n")
    # Check Multiple Weights
    acc_min_loss = []
    acc_min_loss.append(test_acc)
    index = 0
    prev_acc = 100
    with torch.no_grad():
        model = torchvision.models.resnet18(pretrained=False).to(device)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        model.maxpool = nn.Identity().to(device)
        model.fc = nn.Linear(model.fc.in_features, args.num_of_class).to(device)

        if mode == 'standard':
            model.load_state_dict(
                torch.load(args.save_path + args.standard_model_name, map_location=torch.device('cpu')))
        elif mode == 'pruned':
            model.load_state_dict(
                torch.load(args.save_path + args.pruned_model_name, map_location=torch.device('cpu')))
        elif mode == 'fault_aware':
            model.load_state_dict(
                torch.load(args.save_path + args.standard_noisy_model_name, map_location=torch.device('cpu')))

        model_before = copy.deepcopy(model)

        for ls, param, idx, layer_name, param_type, new_weight_value in loss_sorted:
            model_module = dict(model.named_modules())[layer_name]
            index += 1

            if isinstance(model_module, (torch.nn.Conv2d, torch.nn.Linear)) and param_type == 'weight':
                model_weight = getattr(model_module, param_type)
                if len(model_weight.shape) == 4:
                    channel = int(idx // (model_weight.shape[1] * model_weight.shape[2] * model_weight.shape[3]))
                    temp = int(idx - channel * (model_weight.shape[1] * model_weight.shape[2] * model_weight.shape[3]))
                    ic = int(temp // (model_weight.shape[2] * model_weight.shape[3]))
                    temp = int(temp - (ic * (model_weight.shape[2] * model_weight.shape[3])))
                    i = int(temp // model_weight.shape[3])
                    j = int(temp % model_weight.shape[3])
                    original_float_value = model_weight[channel][ic][i][j].clone().detach()
                else:
                    channel = int(idx // model_weight.shape[1])
                    i = int(idx % model_weight.shape[1])
                    original_float_value = model_weight[channel][i].clone().detach()

                # update weight
                weights = model_module.weight.data.view(-1)
                weights[idx] = new_weight_value
                model_module.weight.data = weights.view(model_module.weight.data.size())
                # print('------------------------------------------------------')
                model.eval()
                test_total = 0
                test_correct = 0
                ########################################
                changes = compare_model_weights(model_before, model)
                chg = 0
                for key, (before, after) in changes.items():
                    temp = before - after
                    chg += torch.count_nonzero(temp)
                print(f"Total change = {chg}")
                # input()
                ########################################
                for _, (data, target) in enumerate(test_loader):
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()

                acc_min_loss.append(100.0 * test_correct / test_total)
                print(f'The accuracy for Changing the weight {index} from the original value = {original_float_value} '
                      f'to the new value = {new_weight_value} is : {acc_min_loss[index - 1]: .2f}')

    attack_time = time.time() - start_time
    if mode == 'standard':
        torch.save(acc_min_loss, args.save_path + '/Attack_Standard_Acc_{0}_{1}.pt'.format(attack_type, round_num))
        torch.save(loss_sorted, args.save_path + '/Sorted_weight_standard_{0}_{1}.pt'.format(attack_type, round_num))
    elif mode == 'pruned':
        torch.save(acc_min_loss, args.save_path + '/Attack_pruned_Acc_{0}_{1}.pt'.format(attack_type, round_num))
        torch.save(loss_sorted, args.save_path + '/Sorted_weight_pruned_{0}_{1}.pt'.format(attack_type, round_num))
    elif mode == 'fault_aware':
        torch.save(acc_min_loss, args.save_path + '/Attack_Standard_Noisy_Acc_{0}_{1}.pt'.format(attack_type,
                                                                                                 round_num))
        torch.save(loss_sorted, args.save_path + '/Sorted_weight_Noisy_{0}_{1}.pt'.format(attack_type, round_num))

    return attack_time


def prune_model(test_loader, device, args):
    model = torchvision.models.resnet18(pretrained=False).to(device)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    model.maxpool = nn.Identity().to(device)
    model.fc = nn.Linear(model.fc.in_features, args.num_of_class).to(device)
    model.load_state_dict(torch.load(args.save_path + args.standard_model_name, map_location=torch.device('cpu')))

    model_pruned = copy.deepcopy(model)
    parameters_to_prune = []
    for module_name, module in model_pruned.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=args.amount
    )

    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    # Evaluate pruned model
    pruned_accuracy = test_model(model_pruned, test_loader, device)

    # Save pruned model
    torch.save(model.state_dict(), args.save_path + args.pruned_model_name)

    return model_pruned, pruned_accuracy


def calculate_sparsity(model):
    total_zeros = 0
    total_elements = 0
    layer_sparsity = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight.data
            num_zeros = torch.sum(weight == 0).item()
            num_elements = weight.numel()
            sparsity = 100.0 * num_zeros / num_elements
            layer_sparsity[name] = sparsity
            total_zeros += num_zeros
            total_elements += num_elements

    total_sparsity = 100.0 * total_zeros / total_elements
    return layer_sparsity, total_sparsity


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # define model parameters
    parser = argparse.ArgumentParser(description='Resnet18 with CIFAR100 Dataset!')
    parser.add_argument('--device', type=bool, default=True, help='Train the model on GPU or CPU')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='Destination Folder to Save the Result.')
    parser.add_argument('--batch_size', type=int, default=128, help='Set the Batch Size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Set the Learning Rate.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Set the Gamma Value.')
    parser.add_argument('--steps', type=int, default=100, help='Set the Steps of Training.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Set the Gamma Value.')
    parser.add_argument('--epochs', type=int, default=200, help='Set Number of Epochs.')
    parser.add_argument('--num_of_class', type=int, default=100, help='Number of Output Classes.')
    parser.add_argument('--fixed', type=int, default=1, help='Number of digits for integer part.')
    parser.add_argument('--total', type=int, default=8, help='Number of digits for integer part.')
    parser.add_argument('--fraction', type=int, default=7, help='Number of digits for fraction parts.')
    parser.add_argument('--start_noise_point', type=int, default=-1,
                        help='The number of epochs, we need to start applying noise.')
    parser.add_argument('--num_of_worker', type=int, default=2, help='Number of Workers.')
    parser.add_argument('--num_of_weights', type=int, default=100, help='Number of Weights to Change.')
    parser.add_argument('--standard_model_name', type=str, default='Resnet18_Standard.pt',
                        help='The name of Original Model')
    parser.add_argument('--pruned_model_name', type=str, default='Resnet18_pruned.pt',
                        help='The name of Original Model')
    parser.add_argument('--standard_noisy_model_name', type=str, default='Resnet18_Standard_with_Noise.pt',
                        help='The name of standard model with Noisy training')
    parser.add_argument('--GPU_number', type=int, default=0,
                        help='The number of GPU that you want to select.')
    parser.add_argument('--round_number', type=int, default=1,
                        help='Number of rounds to run the application.')
    parser.add_argument('--start_point', type=int, default=1,
                        help='The start point for saving data in each round')
    parser.add_argument('--model', type=str, default='ResNet18',
                        help='The Model name')
    parser.add_argument('--pruning_percentage', type=float, default=0.7, help='The percentage of pruning')
    parser.add_argument('--prune_step', type=int, default=20,
                        help='The steps for doing pruning during the training')
    parser.add_argument('--start_pruning_epoch', type=int, default=100,
                        help='The steps for doing pruning during the training')
    args = parser.parse_args()

    # Select device: GPU or CPU
    cuda_flag = args.device
    device = torch.device('cuda:{0}'.format(args.GPU_number) if cuda_flag else 'cpu')
    print('\nDevice : ', device)
    print('Number of Round: ', args.round_number)
    # Create a Directory
    current_dir = os.getcwd()
    args.save_path = current_dir + '/' + args.save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for rnd in range(args.round_number):
        #random_seed_number = random.randint(1, 10000)
        random_seed_number = 6178
        torch.manual_seed(random_seed_number)

        print(f'Preparing Dataset in Round {rnd}')
        # Dataset Loading
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                 download=True, transform=transform_test)
        # Split the dataset into training and validation sets
        train_size = int(0.9 * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_set, valid_set = random_split(train_dataset, [train_size, valid_size])

        # Create data loaders for training and validation sets
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_of_worker)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_of_worker)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_of_worker)

        print(f'Round Number = {rnd + args.start_point}')
        print(f'Random Number Seed = {random_seed_number}')
        print('Number of Selected Weights : ', args.num_of_weights)

        args.standard_model_name = 'Model_{0}_Standard_{1}.pt'.format(args.model, random_seed_number)
        args.pruned_model_name = 'Model_{0}_Pruned_{1}.pt'.format(args.model, random_seed_number)
        args.standard_noisy_model_name = 'Model_{0}_Standard_Noise_{1}.pt'.format(args.model, random_seed_number)
        print(f'The name of Standard Model is         : {args.standard_model_name}')
        print(f'The name of Pruned Model is           : {args.pruned_model_name}')
        print(f'The name of Standard Model with Noise : {args.standard_noisy_model_name}')
        # ============================================================================================
        #                                    Run Training ResNet20
        # ============================================================================================
        
        print(f'----------------------------------------------------------------------------------------')
        print(f'            Standard Training and Validation in round {rnd + args.start_point} is started ...')
        print(f'----------------------------------------------------------------------------------------')
        original_acc = train_and_validate(train_loader, valid_loader, test_loader, device, args, random_seed_number)

        print(f'----------------------------------------------------------------------------------------')
        print(f'            Standard Pruning in round {rnd + args.start_point} is started ...')
        print(f'----------------------------------------------------------------------------------------')
        pruned_model, pruned_acc = prune_model(test_loader, device, args)

        print(f'----------------------------------------------------------------------------------------')
        print(f'            Compared Models in round {rnd + args.start_point} ...')
        print(f'----------------------------------------------------------------------------------------')
        layer_sparsity, total_sparsity = calculate_sparsity(pruned_model)
        print("Layer-wise Sparsity:")
        for layer, sparsity in layer_sparsity.items():
            print(f"{layer}: {sparsity:.2f}% sparsity")

        print(f"Total Model Sparsity: {total_sparsity:.2f}%")
        print(f'The Original model accuracy on Test dataset: {original_acc:.2f}%')
        print(f'The Pruned model accuracy on Test dataset: {pruned_acc:.2f}%')
        accuracy_drop = original_acc - pruned_acc
        print(f"Accuracy Drop: {accuracy_drop:.2f}%")
        torch.save(layer_sparsity, args.save_path + '/layer_sparsity.pt')
        torch.save(total_sparsity, args.save_path + '/Total_sparsity.pt')
        torch.save(original_acc, args.save_path + '/Original_acc.pt')
        torch.save(pruned_acc, args.save_path + '/Pruned_acc.pt')
        torch.save(accuracy_drop, args.save_path + '/accuracy_drop.pt')
        

        print(f'----------------------------------------------------------------------------------------')
        print(f'      Noisy Training and Validation On Standard Model in round {rnd + args.start_point} is started ...')
        print(f'----------------------------------------------------------------------------------------')
        noisy_acc = fault_aware_train_validation(train_loader, valid_loader, test_loader, device, args,
                                                 random_seed_number, pruning_rate=args.pruning_percentage,
                                                 mode='BitMapped')
        noisy_sparsity_report = calculate_model_sparsity(device, mode='fault_aware')
        
        print(f'----------------------------------------------------------------------------------------')
        print(f'                 Attack on Standard model in round {rnd + args.start_point} is started ...')
        print(f'----------------------------------------------------------------------------------------')
        original_attack_time = attack(valid_loader, test_loader, device, args, random_seed_number, mode='standard',
                                      attack_type='sign')
        pruned_attack_time = attack(valid_loader, test_loader, device, args, random_seed_number, mode='pruned',
                                      attack_type='sign')
        print(f'The total attack time for the original model after 100 bit flipping: {original_attack_time} sec')
        print(f'The total attack time for the pruned model after 100 bit flipping: {pruned_attack_time} sec')
        

        print(f'----------------------------------------------------------------------------------------')
        print(f'            Attack on Noisy Trained Standard model in round {rnd + args.start_point} is started ...')
        print(f'----------------------------------------------------------------------------------------')
        attack(valid_loader, test_loader, device, args, rnd + args.start_point, mode='fault_aware',
               attack_type='sign', attack_scenario='important')

        print(f'----------------------------------------------------------------------------------------')
        print(f'            Random Attack on Standard model in round {rnd + args.start_point} is started ...')
        print(f'----------------------------------------------------------------------------------------')
        attack(valid_loader, test_loader, device, args, rnd + args.start_point, mode='standard',
               attack_type='sign', attack_scenario='random')
