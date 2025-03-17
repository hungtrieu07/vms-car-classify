import os
import pandas

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

num_workers = min(os.cpu_count(), 6)

color_class_names = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']

def load_class_names(path='/kaggle/input/car-model-make-cartype-color/data/devkit/class_names.csv'):
    cn = pandas.read_csv(path, header=None).values.reshape(-1)
    cn = cn.tolist()
    return cn

def load_annotations_v1(path):
    ann = pandas.read_csv(path, header=None).values
    ret = {}
    for idx in range(len(ann)):
        x1, y1, x2, y2, target, imgfn = ann[idx]
        r = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'target': target - 1,
            'filename': imgfn
        }
        ret[idx] = r
    return ret

def load_annotations_v2(path, v2_info):
    ann = pandas.read_csv(path, header=None).values
    ret = {}
    make_codes = v2_info['make'].astype('category').cat.codes
    type_codes = v2_info['model_type'].astype('category').cat.codes
    for idx in range(len(ann)):
        x1, y1, x2, y2, target, imgfn = ann[idx]
        r = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'target': target - 1,
            'make_target': make_codes[target - 1].item(),
            'type_target': type_codes[target - 1].item(),
            'filename': imgfn
        }
        ret[idx] = r
    return ret

def load_annotations_cmmt(path, cmmt_info):
    ann = pandas.read_csv(path, header=None).values
    ret = {}
    make_codes = cmmt_info['make'].astype('category').cat.codes
    type_codes = cmmt_info['model_type'].astype('category').cat.codes
    #print(type(cmmt_info['model_type']))
    #print(pandas.Series(color_class_names))
    color_codes = pandas.Series(color_class_names).astype('category').cat.codes
    for idx in range(len(ann)):
        target, color, imgfn = ann[idx]
        r = {
            'target': target - 1,
            'make_target': make_codes[target - 1].item(),
            'type_target': type_codes[target - 1].item(),
            'color_target': color_codes[color].item(),
            'filename': imgfn
        }
        ret[idx] = r
    return ret

def separate_class(class_names):
    arr = []
    for idx, name in enumerate(class_names):
        splits = name.split(' ')
        make = splits[0]
        model = ' '.join(splits[1:-1])
        model_type = splits[-2]

        if model == 'General Hummer SUV':
            make = 'AM General'
            model = 'Hummer SUV'
        if model == 'Integra Type R':
            model_type = 'Type-R'
        if model_type == 'Z06' or model_type == 'ZR1':
            model_type = 'Convertible'
        if 'SRT' in model_type:
            model_type = 'SRT'
        if model_type == 'IPL':
            model_type = 'Coupe'

        year = splits[-1]
        arr.append((idx, make, model, model_type, year))

    arr = pandas.DataFrame(arr, columns=['target', 'make', 'model', 'model_type', 'year'])
    return arr

class CarsDatasetV1(Dataset):
    def __init__(self, imgdir, anno_path, transform, size):
        self.annos = load_annotations_v1(anno_path)
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]

        target = r['target']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)

            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)

        return img, target

class CarsDatasetV2(Dataset):
    def __init__(self, imgdir, anno_path, transform, size):
        self.class_names = load_class_names()
        self.v2_info = separate_class(self.class_names)
        self.annos = load_annotations_v2(anno_path, self.v2_info)
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]

        target = r['target']
        make_target = r['make_target']
        type_target = r['type_target']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)

            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)

        return img, target, make_target, type_target
    
class CMMTDataset(Dataset):
    def __init__(self, imgdir, anno_path, transform, size):
        self.class_names = load_class_names()
        self.cmmt_info = separate_class(self.class_names)
        self.annos = load_annotations_cmmt(anno_path, self.cmmt_info)
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]

        target = r['target']
        make_target = r['make_target']
        type_target = r['type_target']
        color_target = r['color_target']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)

            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)

        return img, target, make_target, type_target, color_target

def prepare_loader(config, transform_T=1,
                   train_annopath='data/devkit/cars_train_annos.csv',
                   test_annopath='data/devkit/cars_test_annos_withlabels.csv',
                   train_imgdir = 'data/cars_train',
                   test_imgdir = 'data/cars_test'
                   ):

    if transform_T == 1:
        train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                    ])    
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                    ])
    elif transform_T == 2:
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
        test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    if config['version'] == 1:
        CarsDataset = CarsDatasetV1
    elif config['version'] == 2 or config['version'] == 3:
        CarsDataset = CarsDatasetV2
    elif config['version'] == 4:
        CarsDataset = CMMTDataset

    train_dataset = CarsDataset(train_imgdir, train_annopath, train_transform, config['imgsize'])
    test_dataset = CarsDataset(test_imgdir, test_annopath, test_transform, config['imgsize'])

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=False,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_batch_size'],
                             shuffle=False,
                             pin_memory=False,
                             num_workers=num_workers)

    return train_loader, test_loader