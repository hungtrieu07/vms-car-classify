import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

import pandas
import os
import json
import argparse
import time

import network
import data_process
import test

def train_cmmmt(ep, model, optimizer, lr_scheduler, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0
    color_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target, make_target, type_target, color_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = make_target.to(device)
        type_target = type_target.to(device)
        color_target = color_target.to(device)

        optimizer.zero_grad()

        pred, make_pred, type_pred, color_pred = model(data)

        loss_main = F.cross_entropy(pred, target)
        loss_make = F.cross_entropy(make_pred, make_target)
        loss_type = F.cross_entropy(type_pred, type_target)
        loss_color = F.cross_entropy(color_pred, color_target)

        loss = loss_main + config['make_loss'] * loss_make + config['type_loss'] * loss_type + config['color_loss'] * loss_color
        loss.backward()

        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        type_acc = type_pred.max(1)[1].eq(type_target).float().mean()
        color_acc = color_pred.max(1)[1].eq(color_target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        make_acc_meter += make_acc.item()
        type_acc_meter += type_acc.item()
        color_acc_meter += color_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} '
              f'Make: {make_acc_meter / i:.4f} '
              f'Type: {type_acc_meter / i:.4f} '
              f'Color: {color_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')
        
    lr_scheduler.step()
    
    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)
    type_acc_meter /= len(train_loader)
    color_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_make_acc': make_acc_meter,
        'train_type_acc': type_acc_meter,
        'train_color_acc': color_acc_meter,
        'train_time': elapsed
    }

    return trainres

def train_v2(ep, model, optimizer, lr_scheduler, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target, make_target, type_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = make_target.to(device)
        type_target = type_target.to(device)

        optimizer.zero_grad()

        pred, make_pred, type_pred = model(data)

        loss_main = F.cross_entropy(pred, target)
        loss_make = F.cross_entropy(make_pred, make_target)
        loss_type = F.cross_entropy(type_pred, type_target)

        loss = loss_main + config['make_loss'] * loss_make + config['type_loss'] * loss_type
        loss.backward()

        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        type_acc = type_pred.max(1)[1].eq(type_target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        make_acc_meter += make_acc.item()
        type_acc_meter += type_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} '
              f'Make: {make_acc_meter / i:.4f} '
              f'Type: {type_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')
    
    lr_scheduler.step()

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)
    type_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_make_acc': make_acc_meter,
        'train_type_acc': type_acc_meter,
        'train_time': elapsed
    }

    return trainres

def train_v1(ep, model, optimizer, lr_scheduler, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0
    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        pred = model(data)

        loss = F.cross_entropy(pred, target)
        loss.backward()

        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} ({elapsed:.2f}s)', end='\r')
    
    lr_scheduler.step()
    
    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_time': elapsed
    }

    return trainres

def get_exp_dir(config):
    exp_dir = f'logs/{config["arch"]}_{config["imgsize"][0]}_{config["epochs"]}_v{config["version"]}'

    if config['finetune']:
        exp_dir += '_finetune'

    os.makedirs(exp_dir, exist_ok=True)

    exps = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    files = set(map(int, exps))
    if len(files):
        exp_id = min(set(range(1, max(files) + 2)) - files)
    else:
        exp_id = 1

    exp_dir = os.path.join(exp_dir, str(exp_id))
    os.makedirs(exp_dir, exist_ok=True)

    json.dump(config, open(exp_dir + '/config.json', 'w'))

    return exp_dir

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'batch_size': args.batch_size,
        'test_batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'epochs': args.epochs,
        'imgsize': (args.imgsize, args.imgsize),
        'arch': args.arch,
        'version': args.version,
        'make_loss': args.make_loss,
        'type_loss': args.type_loss,
        'color_loss': args.color_loss,    
        'finetune': args.finetune,
        'path': args.path
    } 

    exp_dir = get_exp_dir(config)
    classes = data_process.load_class_names()
    info = data_process.separate_class(classes)
    num_classes = len(classes)
    num_makes = len(info['make'].unique())
    num_types = len(info['model_type'].unique())
    num_colors = 9

    # Construct the model
    model = network.construct_model(config, num_classes, num_makes, num_types, num_colors)

    # Move model to GPU and enable multi-GPU support if available
    if torch.cuda.is_available():
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)  # Wrap the model for multi-GPU training
    else:
        model = model.to(device)  # Single CPU or single GPU

    # Optimizer uses the wrapped model's parameters
    optimizer = optim.SGD(model.parameters(),
                          lr=config['lr'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)

    train_loader, test_loader = data_process.prepare_loader(config, transform_T=1,
                                                            train_annopath='/kaggle/input/car-model-make-cartype-color/data/devkit/cmmt_train.csv',
                                                            test_annopath='/kaggle/input/car-model-make-cartype-color/data/devkit/cmmt_test.csv',
                                                            train_imgdir='/kaggle/input/car-model-make-cartype-color/data/train_cropped',
                                                            test_imgdir='/kaggle/input/car-model-make-cartype-color/data/test_cropped')

    best_acc = 0
    res = []

    if config['version'] == 1:
        train_fn = train_v1
        test_fn = test.test_v1
    elif config['version'] == 2 or config['version'] == 3:
        train_fn = train_v2
        test_fn = test.test_v2
    elif config['version'] == 4:
        train_fn = train_cmmmt
        test_fn = test.test_cmmt

    for ep in range(1, config['epochs'] + 1):
        train_res = train_fn(ep, model, optimizer, lr_scheduler, train_loader, device, config)
        val_res = test_fn(model, test_loader, device, config)
        train_res.update(val_res)

        if best_acc < val_res['val_acc']:
            best_acc = val_res['val_acc']
            # Save the underlying model's state dict if using DataParallel
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), exp_dir + '/best.pth')
            else:
                torch.save(model.state_dict(), exp_dir + '/best.pth')

        res.append(train_res)
    
    print(f'Best accuracy: {best_acc:.4f}')
    res = pandas.DataFrame(res)
    res.to_csv(exp_dir + '/history.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and finetuning script for Cars classification task')

    # training arg
    parser.add_argument('--batch-size', default=32, type=int,
                        help='training batch size (default: 32)')
    parser.add_argument('--epochs', default=40, type=int,
                        help='training epochs (default: 40)')
    parser.add_argument('--arch', default='resnext50', choices=['resnext50',
                                                                'resnet34',
                                                                'mobilenetv2'],
                        help='Architecture (default: resnext50)')
    parser.add_argument('--imgsize', default=400, type=int,
                        help='Input image size (default: 400)')
    parser.add_argument('--version', default=1, type=int, choices=[1, 2, 3, 4],
                        help='Classification version (default: 1)\n'
                             '1. Cars Model only\n'
                             '2. Cars Model + Make + Car Type\n'
                             '4. Cars Model + Make + Car Type + Color')	
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='whether to finetune from 400x400 to 224x224 (default: False)')
    parser.add_argument('--path',
                        help='required if it is a finetune task (default: None)')

    # optimizer arg
    parser.add_argument('--lr', default=0.01, type=float,
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', default=0.0001, type=float,
                        help='SGD weight decay (default: 0.0001)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum (default: 0.9)')

    # multi-task learning arg
    parser.add_argument('--make-loss', default=0.2, type=float,
                        help='loss$_{make}$ lambda')
    parser.add_argument('--type-loss', default=0.2, type=float,
                        help='loss$_{type}$ lambda')
    parser.add_argument('--color-loss', default=0.2, type=float,
                        help='loss$_{color}$ lambda')

    args = parser.parse_args()
    main(args)