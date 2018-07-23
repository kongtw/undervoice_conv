# -*- coding: utf-8 -*-
# !/usr/bin/env python2.7
import wavsegment as wg
import commands
import os
import ufuncs as uf
import numpy as np
import pymysql

def get_wavchunk(wav_name, database_cursor, database_conn, wav_tmp_path, chunk_tmp, hash_str, return_dict):
    try:
        wav_8k_name = wav_tmp_path + wav_name.split('/')[-1].split('.')[0] + '.wav'
        sox_cmd = 'sox -r 6k ' + wav_name + ' -r 8k ' + wav_8k_name
        (status, output) = commands.getstatusoutput(sox_cmd)
        print '%s: %s' % (wav_8k_name, output)
        wav_index = wav_name.split('/')[-1].split('.')[0]
        kf_id = wav_name.split('/')[-1].split('.')[3]
        audio, sample_rate = wg.read_wave(wav_8k_name)
        vad = wg.webrtcvad.Vad(int(3))
        # 每个frame持续30ms
        frames = wg.frame_generator(30, audio, sample_rate)
        frames = list(frames)
        begin_time_list = []
        end_time_list = []
        segments = wg.vad_collector(sample_rate, 30, 300, vad, frames)
        chunk_list = []
        for i, segment in enumerate(segments):
            path = chunk_tmp + '%s_chunk_voice-%004d.wav' % (wav_index, i)
            chunk_list.extend([path])
            wg.write_wave(path, segment[0], sample_rate)
            begin_time_list.append(30 * segment[1] / 1000.0)
            end_time_list.append(30 * (segment[2] + 1) / 1000.0)
        count = 0
        print 'inserting %s' % wav_8k_name
        for i, wav_chunk in enumerate(chunk_list):
            try:
                if count == 0:
                    value_one = [(wav_8k_name, os.path.abspath(wav_chunk), str(0), str(wav_index), str('-1'), str(0.0),
                                  str(kf_id), str(-1), hash_str, str(begin_time_list[i]), str(end_time_list[i]))]
                    database_cursor.executemany(
                        "INSERT INTO runtime_wav_prob VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", value_one)
                    database_conn.commit()
                else:
                    value_one = [(wav_8k_name, os.path.abspath(wav_chunk), str(1), str(wav_index), str('-1'), str(0.0),
                                  str(kf_id), str(-1), hash_str, str(begin_time_list[i]), str(end_time_list[i]))]
                    database_cursor.executemany(
                        "INSERT INTO runtime_wav_prob VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", value_one)
                    database_conn.commit()
            except BaseException, e:
                print 'insert db error %s' % str(e)
                return_dict['db_insert_error'] = os.getpid()
                exit(-1)
                return -1
            count += 1
        return 0
    except BaseException, e:
        exit(-1)
        return -1

#############################
###保存静音部分做信噪比计算###
############################
def get_wavchunk_all(wav_name, database_cursor, database_conn, wav_tmp_path, chunk_tmp, hash_str, return_dict):
    try:
        wav_8k_name = wav_tmp_path + wav_name.split('/')[-1].split('.')[0] + '.wav'
        sox_cmd = 'sox -r 6k ' + wav_name + ' -r 8k ' + wav_8k_name
        (status, output) = commands.getstatusoutput(sox_cmd)
        print '%s: %s' % (wav_8k_name, output)
        wav_index = wav_name.split('/')[-1].split('.')[0]
        kf_id = wav_name.split('/')[-1].split('.')[3]
        audio, sample_rate = wg.read_wave(wav_8k_name)
        vad = wg.webrtcvad.Vad(int(2))
        # 每个frame持续30ms
        frames = wg.frame_generator(30, audio, sample_rate)
        frames = list(frames)
        begin_time_list = []
        end_time_list = []
        segments = wg.vad_segment_c(sample_rate, 30, 300, vad, frames)
        chunk_list = []
        voice_flag_list = []
        speach_en_c = []
        speach_sample_c = []
        for i, segment in enumerate(segments):
            path = chunk_tmp + '%s_chunk_voice-%004d.wav' % (wav_index, i)
            wg.write_wave(path, segment[0][1:], sample_rate)
            # 检测到的录音中能量
            speach_en = uf.speech_energy(path)
            speach_en_c.extend([speach_en])
            # 录音信号段的采样个数
            sample_num = uf.speech_samples_count(path)
            speach_sample_c.extend([sample_num])
            chunk_list.extend([path])
            begin_time_list.append(30 * segment[1] / 1000.0)
            end_time_list.append(30 * segment[2] / 1000.0)
            if segment[0][0] == '1':
                voice_flag_list.extend(['1'])
            else:
                voice_flag_list.extend(['0'])
        count = 0
        print 'inserting %s' % wav_8k_name
        for i, wav_chunk in enumerate(chunk_list):
            try:
                if count == 0:
                    value_one = [
                        (wav_name, os.path.abspath(wav_chunk), str('-1'), str(speach_en_c[i]),
                         str(speach_sample_c[i]),
                         str('-1'), str('-1'), str('-1'), str('-1'), str('-1'), str('-1'), str('-1'), str('-1'),
                         str('-1'), str('-1'),
                         str('-1'), str('-1'), str(1), str(wav_index), str('-1'), str('-1'), str(kf_id),
                         str(begin_time_list[i]), str(end_time_list[i]), str(voice_flag_list[i]), str(hash_str),
                         str(-1), str(-1), str(-1), str(-1), str(-1))]
                    database_cursor.executemany(
                        "INSERT INTO runtime_prob VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        value_one)
                    database_conn.commit()
                else:
                    value_one = [
                        (wav_name, os.path.abspath(wav_chunk), str('-1'), str(speach_en_c[i]),
                         str(speach_sample_c[i]),
                         str('-1'), str('-1'), str('-1'), str('-1'), str('-1'), str('-1'), str('-1'), str('-1'),
                         str('-1'), str('-1'),
                         str('-1'), str('-1'), str(0), str(wav_index), str('-1'), str('-1'), str(kf_id),
                         str(begin_time_list[i]), str(end_time_list[i]), str(voice_flag_list[i]), str(hash_str),
                         str(-1), str(-1), str(-1), str(-1), str(-1))]
                    database_cursor.executemany(
                        "INSERT INTO runtime_prob VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        value_one)
                    database_conn.commit()
            except BaseException, e:
                print 'insert db error %s' % str(e)
                return_dict['db_insert_error'] = os.getpid()
                exit(-1)
                return -1
            count += 1
        return 0
    except BaseException, e:
        exit(-1)
        return -1


def cal_wav_snr(wav_name, database_cursor, database_conn, hash_str):
    sql_str = 'select distinct chunk_name, speech_en, samples_num, voice_flag from runtime_prob where wav_conver_name = "%s" and taskkey = "%s";' % (
        wav_name, hash_str)
    database_cursor.execute(sql_str)
    f_list = database_cursor.fetchall()
    chunk_name = []
    speech_en = []
    samples_num = []
    voice_flag = []
    for f_one in f_list:
        chunk_name.extend([f_one[0]])
        speech_en.extend([f_one[1]])
        samples_num.extend([f_one[2]])
        voice_flag.extend([f_one[3]])
    if len(chunk_name) < 10:
        value_one = [(wav_name, hash_str)]
        database_cursor.executemany(
            "UPDATE runtime_prob SET use_flag = 0 WHERE wav_conver_name=%s and taskkey=%s",  value_one)
        database_conn.commit()
        return -1
    else:
        index_v = [i for i in range(0, len(voice_flag)) if voice_flag[i] == 1]
        index_s = [i for i in range(0, len(voice_flag)) if voice_flag[i] == 0]
        speech_en_s_v = np.sum(np.asarray(speech_en)[index_v])
        speech_en_s_s = np.sum(np.asarray(speech_en)[index_s])
        samples_num_s_v = np.sum(np.asarray(samples_num)[index_v])
        samples_num_s_s = np.sum(np.asarray(samples_num)[index_s])

        voice_av_en = speech_en_s_v / samples_num_s_v
        noise_av_en = speech_en_s_s / samples_num_s_s

        snr_av = (voice_av_en - noise_av_en) / noise_av_en

        if snr_av >= 4:
            value_one = [(wav_name, hash_str)]
            database_cursor.executemany(
                "UPDATE runtime_prob SET use_flag = 1 WHERE wav_conver_name=%s and taskkey=%s", value_one)
            database_conn.commit()
        else:
            value_one = [(wav_name, hash_str)]
            database_cursor.executemany(
                "UPDATE runtime_prob SET use_flag = 0 WHERE wav_conver_name=%s and taskkey=%s", value_one)
            database_conn.commit()

        return snr_av

if __name__ == "__main__":
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                           charset='utf8')
    cursor = conn.cursor()
    hash_str = '2aaceaf407e331f84ab8e5af052017c9'
    sql_str = 'select distinct wav_conver_name, wav_index from runtime_prob;'
    cursor.execute(sql_str)
    wav_name_list = cursor.fetchall()
    snr_av = []
    for wav_name in wav_name_list:
        snr_av_one = cal_wav_snr(wav_name[0], cursor, conn, hash_str)
        if snr_av_one != -1:
            snr_av.extend([snr_av_one])
        else:
            pass
    print(len(snr_av))
    print(snr_av)
    print np.mean(np.asarray(snr_av))
    np.save('snr_av_mean.npy', np.mean(np.asarray(snr_av)))
    np.save('snr_av.npy', snr_av)
