import os
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import numpy as np


s_dir = './source'
t_dir = './target'

source_name = os.listdir(s_dir)
L = len(source_name)
for l in range(L):
    print(L - l)
    s = Image.open('{}/{}.jpg'.format(s_dir, l)).convert('RGB')
    t = Image.open('{}/{}.png'.format(t_dir, l)).convert('L')

    s = s.resize((640, 480))
    t = t.resize((640, 480))

    s.save('G:\\raspi_cansat\source\{}.jpg'.format(l))
    t.save('G:\\raspi_cansat\\target\{}.png'.format(l))


    # Image.fromarray(s).convert('RGB').save('./source2/{}.jpg'.format(l + L))
    # Image.fromarray(t).convert('L').save('./target2/{}.png'.format(l + L))
