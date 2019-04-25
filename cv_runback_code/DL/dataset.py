import os
from PIL import Image
import numpy as np
import pickle


def make_dataset(source_dir="./s", target_dir="./t"):
    h, w = 256, 256
    source = []
    target = []
    source_name = os.listdir(source_dir)

    L = len(source_name)
    for l in range(L):
        print(L - l)

        s = Image.open('{}/{}.jpg'.format(source_dir, l)).convert('RGB')
        t = Image.open('{}/{}.png'.format(target_dir, l)).convert('L')  # gray scale

        s = np.array(s, dtype=np.uint8).transpose((2, 0, 1))
        t = np.array(t, dtype=np.uint8).reshape((1, h, w))

        source.append(s)
        target.append(t)
    print('loaded')
    return source, target


def resize_data():
    source_dir = "./source"
    target_dir = "./target"
    source_name = os.listdir(source_dir)
    L = len(source_name)
    for l in range(11448, 11660):
        print(211 - l)

        s = Image.open('{}/{}.jpg'.format(source_dir, l)).convert('RGB')
        t = Image.open('{}/{}.png'.format(target_dir, l)).convert('L')  # gray scale

        s.resize((256, 256)).save('./{}/{}.jpg'.format('s', int(l - 11448 + 211)))
        t.resize((256, 256)).save('./{}/{}.png'.format('t', int(l - 11448 + 211)))


if __name__ == '__main__':
    source, target = make_dataset('./s', './t')
    with open('source.pickle', 'wb') as f:
        pickle.dump(source, f)

    with open('target.pickle', 'wb') as f:
        pickle.dump(target, f)
    # resize_data()
