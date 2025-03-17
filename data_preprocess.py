import cv2
import pandas

COLS = ['x1', 'y1', 'x2', 'y2', 'label', 'filename']

def crop(csv_path, img_dir, out_dir):
    csvFile = pandas.read_csv(csv_path, names=COLS)
    print(csvFile)

    stopped = True; i = 0

    while True: 
        imgfile = img_dir + csvFile.loc[i, 'filename']
        img = cv2.imread(imgfile)

        img = img[csvFile.loc[i, 'y1']:csvFile.loc[i, 'y2'], csvFile.loc[i, 'x1']:csvFile.loc[i, 'x2']]
        filename = out_dir + csvFile.loc[i, 'filename']
        cv2.imwrite(filename, img)
        i += 1
        if i == len(csvFile): break

import os
import torch
from torchvision import transforms
from PIL import Image

color_class_names = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def transform_image_for_infer(src_path):
    src_image = cv2.imread(src_path)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    if src_image is None: return
    dst_img = cv2.resize(src=src_image, dsize=(224, 224))
    img = Image.fromarray(dst_img)
    image = data_transforms['valid'](img).float()
    image = torch.Tensor(image)
    return image.unsqueeze(0)
#    return image.unsqueeze(0).cuda() # if cuda is available

def pred_bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j][0] < arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def predict_color(src, model, top:int=1):
    image = transform_image_for_infer(src)
    if image is None: return
    model.eval()
    preds = model(image)
    pred_list = preds.tolist()[0]
    pred_list2 = []
    for index, item in enumerate(pred_list):
        pred_list2.append((item, index))

    pred_bubbleSort(pred_list2)
    pred_list2 = pred_list2[:top]
    pred_list3 = []

    for item in pred_list2:
        pred_list3.append((color_class_names[item[1]], item[0]))

    return pred_list3

def predict_color_folder(folder, model, top:int=1):
    infer_list = os.listdir(folder)
    results = []
    for img in infer_list:
        src = folder + img
        results.append((img, predict_color(src, model, top)))
    return results

def color_results2csv(results, out_csv):
    df = pandas.DataFrame(results, columns=['Color', 'Image'])
    df.to_csv(out_csv, index=False)

def generate_new_csv(file):
    df = pandas.read_csv(file, names=['x1', 'y1', 'x2', 'y2', 'label', 'filename'])
    print(df[['label', 'filename']])
    new_df = df[['label', 'filename']]
    new_df.to_csv('./data/devkit/cars_test_annos_new.csv', index=False)

def generate_cmmt_csv(file1, file2):
    df1 = pandas.read_csv(file1)
    print(df1)
    df2 = pandas.read_csv(file2)
    print(df2)

    result_list = []; i = 0

    for color, car in zip(df1['Color'], df2['label']):
        for index in range(0, len(color_class_names)):
            if color_class_names[index] == color:
                color = index
                break
        result_list.append((car, color, df2.loc[i, 'filename']))
        i += 1

    df = pandas.DataFrame(result_list, columns=['label', 'color', 'filename'])
    print(df)
    df.to_csv('./data/devkit/cmmt_test.csv', index=False)

if __name__ == '__main__':
    #generate_new_csv('./data/devkit/cars_test_annos_withlabels.csv')
    generate_cmmt_csv('./color_test.csv', './data/devkit/cars_test_annos_new.csv')
    pass