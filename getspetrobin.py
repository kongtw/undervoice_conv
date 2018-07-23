from PIL import Image
import numpy as np
import glob
import pymysql
import struct
import sys
import random


def build_bin(spectrogram_name):
    f = open('./predictdata/data_predict.bin', 'wb')
    im = Image.open(spectrogram_name).resize((64, 64))
    im = np.array(im)
    r = im[:, :].flatten()
    g = im[:, :].flatten()
    b = im[:, :].flatten()
    if len(r) != 64 * 64:
        print(len(r))
        sys.exit()
    label = [int(6)]
    print label
    alldata = list(label) + list(r) + list(g) + list(b)
    out = np.array(alldata, np.uint8)
    myfmt = 'B' * len(out)
    bin = struct.pack(myfmt, *out)
    f.write(bin)
    f.close()


if __name__ == '__main__':
    build_bin()
