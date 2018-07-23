from PIL import Image
import numpy as np
import glob
import pymysql
import struct
import sys
import random

conn = pymysql.connect('127.0.0.1', 'root', '123456', 'undervoice_conv')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM ampli_wav')
for row in cursor:
    global spectrogram_num
    spectrogram_num = row[0]

cursor.execute('SELECT spec_name FROM ampli_wav')

alldata = []
batch_file_num = spectrogram_num // 7
batch_file_counter = 1
image_count = 0
newfileflag = False
f = open('./inputdata/ampli_spectro_2_batches_bin_color/data_batch_1.bin', 'ab')
rows = list(cursor.fetchall())
rows_random = []
for row in rows:
    rows_random.extend([row[0]])
random.shuffle(rows_random)
for row in rows_random:

    if image_count >= batch_file_num * batch_file_counter:
        batch_file_counter += 1
        print('batch_file_counter:' + str(batch_file_counter))
        newfileflag = True

    if newfileflag:
        f.close()
        batch_file_name = './inputdata/ampli_spectro_2_batches_bin_color/data_batch_%d.bin' % batch_file_counter
        f = open(batch_file_name, 'ab')
        newfileflag = False

    image_count += 1
    imagefile = row
    im = Image.open(imagefile).resize((64, 64))
    im = np.array(im)
    r = im[:, :].flatten()
    g = im[:, :].flatten()
    b = im[:, :].flatten()
    if len(r) != 64 * 64:
        print(len(r))
        sys.exit()
    label = [int(imagefile.split('/')[-1].split('.')[0].split('_')[0])]
    print label
    alldata = list(label) + list(r) + list(g) + list(b)
    out = np.array(alldata, np.uint8)
    myfmt = 'B' * len(out)
    bin = struct.pack(myfmt, *out)
    f.write(bin)

f.close()
cursor.close()
conn.close()
