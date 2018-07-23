import glob
import shutil
import os
from multiprocessing import Pool

def main():
    process_num = 1
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(spk_reg, args=())
    p.close()
    p.join()
    print 'all spk sp subprocesses done'

def spk_reg():
    cfg_dict = {}
    cfg_dict['NormFeat'] = 'NormFeat_energy_SPro'
    cfg_dict['NormFeat_spro'] = 'NormFeat_SPro'
    cfg_dict['EnergyDetector'] = 'EnergyDetector_SPro'
    cfg_dict['ivExtractor'] = 'ivExtractor_fast_l'
    cfg_dict['IvTest'] = 'ivTest_WCCN_Cosine_l'
    cfg_dict['Norm'] = 'ivNorm'
    cfg_dict['Norm_cfg'] = 'ivNorm'
    cfg_dict['trainModel'] = 'trainModel'
    rt_cfg_dir = '/home/dell/runtimedata/spk_test_data/cfg_rt/'
    cfg_dir = './cfg/'
    rt_ndx_dir = '/home/dell/runtimedata/spk_test_data/ndx_rt/'
    ndx_dir = './ndx/'

    norm_cfg_name = cfg_dict['NormFeat'] + '_' + str(os.getpid()) + '.cfg'
    dest_norm_cfg = rt_cfg_dir + norm_cfg_name
    src_cfg = cfg_dir + cfg_dict['NormFeat'] + '.cfg'
    shutil.copyfile(src_cfg, dest_norm_cfg)

    norm_spro_cfg_name = cfg_dict['NormFeat_spro'] + '_' + str(os.getpid()) + '.cfg'
    dest_norm_spro_cfg = rt_cfg_dir + norm_spro_cfg_name
    src_cfg = cfg_dir + cfg_dict['NormFeat_spro'] + '.cfg'
    shutil.copyfile(src_cfg, dest_norm_spro_cfg)

    en_cfg_name = cfg_dict['EnergyDetector'] + '_' + str(os.getpid()) + '.cfg'
    dest_en_cfg = rt_cfg_dir + en_cfg_name
    src_cfg = cfg_dir + cfg_dict['EnergyDetector'] + '.cfg'
    shutil.copyfile(src_cfg, dest_en_cfg)

    iv_cfg_name = cfg_dict['ivExtractor'] + '_' + str(os.getpid()) + '.cfg'
    dest_iv_cfg = rt_cfg_dir + iv_cfg_name
    src_cfg = cfg_dir + cfg_dict['ivExtractor'] + '.cfg'
    shutil.copyfile(src_cfg, dest_iv_cfg)

    iv_test_name = cfg_dict['IvTest'] + '_' + str(os.getpid()) + '.cfg'
    dest_test_cfg = rt_cfg_dir + iv_test_name
    src_cfg = cfg_dir + cfg_dict['IvTest'] + '.cfg'
    shutil.copyfile(src_cfg, dest_test_cfg)

    iv_norm_cfg_name = cfg_dict['Norm_cfg'] + '_' + str(os.getpid()) + '.cfg'
    dest_iv_norm_cfg = rt_cfg_dir + iv_norm_cfg_name
    src_cfg = cfg_dir + cfg_dict['Norm_cfg'] + '.cfg'
    shutil.copyfile(src_cfg, dest_iv_norm_cfg)

    iv_norm_name = cfg_dict['Norm'] + '_' + str(os.getpid()) + '.ndx'
    dest_norm_ndx = rt_ndx_dir + iv_norm_name
    src_ndx = ndx_dir + cfg_dict['Norm'] + '.ndx'
    shutil.copyfile(src_ndx, dest_norm_ndx)

    iv_train_model = cfg_dict['trainModel'] + '_' + str(os.getpid()) + '.ndx'
    dest_train_model = rt_ndx_dir + iv_train_model
    src_ndx = ndx_dir + cfg_dict['trainModel'] + '.ndx'
    shutil.copyfile(src_ndx, dest_train_model)

    prm_dir = '/home/dell/runtimedata/spk_test_data/prm/'
    lbl_dir = '/home/dell/runtimedata/spk_test_data/lbl/'
    iv_raw = '/home/dell/runtimedata/spk_test_data/iv_raw/'

    ndx_name = rt_ndx_dir + str(os.getpid()) + '_ivExtractor.ndx'
    test_ndx_name = rt_ndx_dir + str(os.getpid()) + '_ivTest_plda_target-seg.ndx'
    spk_num = 1551
    with open(ndx_name, 'w') as ndx_fp:
        with open(test_ndx_name, 'w') as test_ndx_fp:
            delete_list = []
            for wav_name in glob.glob('/home/dell/runtimedata/spk_test_data/pcm/*'):
                print wav_name
                wav_index = wav_name.split('/')[-1].split('.')[0] + '_' + str(os.getpid())
                ndx_fp.writelines(wav_index + ' ' + wav_index + '\n')
                test_ndx_fp.writelines(wav_index)
                for i in xrange(1,spk_num):
                    test_ndx_fp.writelines(' spk%0004d' %i)
                test_ndx_fp.writelines(' spk%0004d' %spk_num + '\n')
                cm_str = 'bin/sfbcep -f 8000 -m -k 0.97 -p19 -n 24 -r 22 -e -D -A -F PCM16  %s ' \
                         '%s/%s.tmp.prm' % (wav_name, prm_dir, wav_index)
                os.system(cm_str)
                delete_list.extend(['%s/%s.tmp.prm' % (prm_dir, wav_index)])

                cm_str = 'bin/NormFeat --config %s --inputFeatureFilename %s ' \
                         '--featureFilesPath  %s' % (dest_norm_cfg, wav_index, prm_dir)
                os.system(cm_str)
                delete_list.extend(['%s/%s.enr.tmp.prm' % (prm_dir, wav_index)])

                if os.path.exists(lbl_dir + wav_index + '.lbl'):
                    os.remove(lbl_dir + wav_index + '.lbl')
                    cm_str = 'bin/EnergyDetector  --config %s --inputFeatureFilename %s ' \
                             '--featureFilesPath  %s  --labelFilesPath  %s' % (dest_en_cfg, wav_index, prm_dir, lbl_dir)
                    os.system(cm_str)
                else:
                    cm_str = 'bin/EnergyDetector  --config %s --inputFeatureFilename %s ' \
                             '--featureFilesPath  %s  --labelFilesPath  %s' % (dest_en_cfg, wav_index, prm_dir, lbl_dir)
                    os.system(cm_str)
                delete_list.extend(['%s/%s.norm.prm' % (prm_dir, wav_index)])

                cm_str = 'bin/NormFeat --config %s --inputFeatureFilename %s ' \
                         '--featureFilesPath %s   --labelFilesPath  %s' % (
                         dest_norm_spro_cfg, wav_index, prm_dir, lbl_dir)
                os.system(cm_str)
                delete_list.extend(['%s/%s.lbl' % (lbl_dir, wav_index)])

    cm_str = 'bin/IvExtractor --config %s --saveVectorFilesPath %s --featureFilesPath %s --labelFilesPath %s ' \
             '--targetIdList %s' % (dest_iv_cfg, iv_raw, prm_dir, lbl_dir, ndx_name)
    os.system(cm_str)
    for iv_one in glob.glob(iv_raw + '/*_%s' % os.getpid() + '.y'):
        delete_list.extend([iv_one])

    cm_str = 'bin/IvTest --config %s --outputFilename %s --ndxFilename %s --targetIdList %s --loadVectorFilesPath %s ' \
             '--testVectorFilesPath %s --backgroundNdxFilename %s' \
             % (dest_test_cfg, '/home/dell/runtimedata/spk_test_data/res/' + str(os.getpid()) + '_res.txt', test_ndx_name, dest_train_model, iv_raw, iv_raw,
                dest_norm_ndx)
    os.system(cm_str)
    for de_one in delete_list:
        os.remove(de_one)


if __name__ == '__main__':
    main()