import argparse
import json
import os
import time

import torch
from torch import nn
import torch.nn.functional as F

import data_process
import network

def test_cmmt(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0
    color_acc_meter = 0
    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, type_target, color_target in test_loader:
            data = data.to(device)
            target = target.to(device)
            make_target = make_target.to(device)
            type_target = type_target.to(device)
            color_target = color_target.to(device)

            pred, make_pred, type_pred, color_pred = model(data)

            loss_main = F.cross_entropy(pred, target)
            loss_make = F.cross_entropy(make_pred, make_target)
            loss_type = F.cross_entropy(type_pred, type_target)
            loss_color = F.cross_entropy(color_pred, color_target)

            loss = loss_main + config['make_loss'] * loss_make + config['type_loss'] * loss_type + config['color_loss'] * loss_color

            acc = pred.max(1)[1].eq(target).float().sum()
            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            type_acc = type_pred.max(1)[1].eq(type_target).float().sum()
            color_acc = color_pred.max(1)[1].eq(color_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            acc_meter += acc.item()
            make_acc_meter += make_acc.item()
            type_acc_meter += type_acc.item()
            color_acc_meter += color_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} '
                  f'Make: {make_acc_meter / runcount:.4f} '
                  f'Type: {type_acc_meter / runcount:.4f} '
                  f'Color: {color_acc_meter / runcount:.4}'
                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        acc_meter /= runcount
        make_acc_meter /= runcount
        type_acc_meter /= runcount
        color_acc_meter /= runcount

    print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'val_make_acc': make_acc_meter,
        'val_type_acc': type_acc_meter,
        'val_color_acc': color_acc_meter,
        'val_time': elapsed
    }

    return valres

def test_v1(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    runcount = 0
    elapsed = 0
    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            pred = model(data)

            loss = F.cross_entropy(pred, target) * data.size(0)
            acc = pred.max(1)[1].eq(target).float().sum()

            loss_meter += loss.item()
            acc_meter += acc.item()
            i += 1
            elapsed = time.time() - start_time
            runcount += data.size(0)

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} ({elapsed:.2f}s)', end='\r')

        print()

        loss_meter /= runcount
        acc_meter /= runcount

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'val_time': elapsed
    }

    print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')

    return valres

def test_v2(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0
    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, type_target in test_loader:
            data = data.to(device)
            target = target.to(device)
            make_target = make_target.to(device)
            type_target = type_target.to(device)

            pred, make_pred, type_pred = model(data)

            loss_main = F.cross_entropy(pred, target)
            loss_make = F.cross_entropy(make_pred, make_target)
            loss_type = F.cross_entropy(type_pred, type_target)

            loss = loss_main + config['make_loss'] * loss_make + config['type_loss'] * loss_type

            acc = pred.max(1)[1].eq(target).float().sum()
            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            type_acc = type_pred.max(1)[1].eq(type_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            acc_meter += acc.item()
            make_acc_meter += make_acc.item()
            type_acc_meter += type_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} '
                  f'Make: {make_acc_meter / runcount:.4f} '
                  f'Type: {type_acc_meter / runcount:.4f} '
                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        acc_meter /= runcount
        make_acc_meter /= runcount
        type_acc_meter /= runcount

    print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'val_make_acc': make_acc_meter,
        'val_type_acc': type_acc_meter,
        'val_time': elapsed
    }

    return valres

def load_weight(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = json.load(open(args.config))
    config['imgsize'] = (args.imgsize, args.imgsize)

    exp_dir = os.path.dirname(args.config)
    modelpath = exp_dir + '/best.pth'

    class_names = data_process.load_class_names()
    num_classes = len(class_names)
    v2_info = data_process.separate_class(class_names)
    num_makes = len(v2_info['make'].unique())
    num_types = len(v2_info['model_type'].unique())

    model = network.construct_model(config, num_classes, num_makes, num_types)
    load_weight(model, modelpath, device)
    model = model.to(device)

    train_loader, test_loader = data_process.prepare_loader(config)
    for batch in test_loader:
        print(f"Test loader batch length: {len(batch)}")
        print(f"Batch contents: {[x.shape if hasattr(x, 'shape') else x for x in batch]}")
        break

    if config['version'] == 1:
        test_fn = test_v1
    else:
        test_fn = test_v2

    test_fn(model, test_loader, device, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script for Cars dataset')

    parser.add_argument('--config', required=True,
                        help='path to config.json')
    parser.add_argument('--imgsize', default=400, type=int,
                        help='img size for testing (default: 400)')

    args = parser.parse_args()

    main(args)