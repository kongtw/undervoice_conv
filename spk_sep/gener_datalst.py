# -*- coding: utf-8 -*-
import glob
import wave
import shutil
import pymysql

def get_datalst():
    with open('./data/rdata.lst', 'w') as rdatalst:
        for wav_name in glob.glob('./data/pcm/spkregister_8k/*'):
            print wav_name
            rdatalst.writelines(wav_name.split('/')[-1].split('.')[0] + '\n')

def get_ubmlst():
    ct = 0
    with open('./lst/UBM.lst', 'w') as ubmlst:
        for wav_name in glob.glob('./data/pcm/spkregister_8k/*'):
            if ct < 300:
                print wav_name
                ubmlst.writelines(wav_name.split('/')[-1].split('.')[0] + '\n')
            else:
                break
            ct += 1

def get_totalndx():
    ct = 0
    with open('./ndx/totalvariability.ndx', 'w') as totalndx:
        for wav_name in glob.glob('./data/pcm/spkregister_8k/*'):
            if ct < 300:
                print wav_name
                totalndx.writelines(wav_name.split('/')[-1].split('.')[0] + '\n')
            else:
                break
            ct += 1

def get_ivndx():
    with open('./ndx/ivExtractor.ndx', 'w') as ivndx:
        for wav_name in glob.glob('./data/pcm/spkregister_8k/*'):
            print wav_name
            ivndx.writelines(wav_name.split('/')[-1].split('.')[0] + ' ' + wav_name.split('/')[-1].split('.')[0] + '\n')

def get_trianmodelndx():
    with open('./ndx/trainModel.ndx', 'w') as trainmodelndx:
        counter = 1
        for wav_name in glob.glob('/home/dell/runtimedata/spk_test_data/iv_raw/*.y'):
            print wav_name
            write_line = 'spk%0004d' % counter + ' ' + wav_name.split('/')[-1].split('.')[0] + '\n'
            trainmodelndx.writelines(write_line)
            counter += 1

def get_ivnormndx():
    with open('./ndx/ivNorm.ndx', 'w') as ivnormndx:
        counter = 0
        wavname_lst = []
        for wav_name in glob.glob('/home/dell/python/undervoice_conv/spk_sep/iv/raw/*.y'):
            print wav_name
            counter += 1
            wavname_lst.extend([wav_name.split('/')[-1].split('.')[0]])
            if counter > 300:
                break
            else:
                pass
            if counter % 3 == 0:
                ivnormndx.writelines(wavname_lst[0] + ' ' + wavname_lst[1] + ' ' + wavname_lst[2] + '\n')
                wavname_lst = []
            else:
                pass

def copy_normiv():
    for iv_name in glob.glob('/home/dell/python/undervoice_conv/spk_sep/iv/raw/*'):
        print iv_name
        iv_index = iv_name.split('/')[-1]
        dest_iv_name = '/home/dell/runtimedata/spk_test_data/iv_raw/' + iv_index
        shutil.copyfile(iv_name, dest_iv_name)

def get_ubmwav():
    ubmwav = wave.open('./data/pcm/spkregister_8k/ubm.wav', 'wb')
    wav_counter = 0
    wav_data = ''
    frame_c = 0
    num_channles = 0
    num_samplewidth = 0
    fr = 0
    for wav_name in glob.glob('/home/dell/下载/客户断句/*'):
        if wav_counter == 0:
            wav_open = wave.open(wav_name, 'rb')
            num_channles = wav_open.getnchannels()
            num_samplewidth = wav_open.getsampwidth()
            fr = wav_open.getframerate()
            frame_n = wav_open.getnframes()
            frame_c += int(frame_n / 500)
            wav_str = wav_open.readframes(int(frame_n / 500))
            print len(wav_str)
            wav_data += wav_str
        elif wav_counter < 18574:
            wav_open = wave.open(wav_name, 'rb')
            frame_n = wav_open.getnframes()
            frame_c += int(frame_n / 500)
            wav_str = wav_open.readframes(int(frame_n / 500))
            print len(wav_str)
            wav_data += wav_str
        else:
            break
        wav_counter += 1
    print len(wav_data)
    ubmwav.setnchannels(num_channles)
    ubmwav.setsampwidth(num_samplewidth)
    ubmwav.setframerate(fr)
    ubmwav.setnframes(frame_c)
    ubmwav.writeframes(wav_data)

def insert_register_kf():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                           charset='utf8')
    cursor = conn.cursor()
    for wav_name in glob.glob('./data/pcm/spkregister_8k/*'):
        kf_id = wav_name.split('/')[-1].split('.')[0]
        value_one = [(kf_id)]
        cursor.executemany(
            "INSERT INTO register_kf VALUES (%s)",
            value_one)
        conn.commit()

if __name__ == '__main__':
    #get_datalst()
    #get_ubmlst()
    #get_totalndx()
    #get_ivndx()
    #get_ivnormndx()
    #copy_normiv()
    #get_ubmwav()
    get_ivnormndx()