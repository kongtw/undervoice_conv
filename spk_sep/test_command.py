import os

cmd_str = 'bin/IvExtractor --config ./cfg_rt/ivExtractor_fast_25520.cfg --saveVectorFilesPath /home/dell/runtimedata/spk_test_data/iv_raw/ ' \
          '--featureFilesPath /home/dell/runtimedata/spk_test_data/prm/ --labelFilesPath /home/dell/runtimedata/spk_test_data/lbl/ --targetIdList /home/dell/runtimedata/spk_test_data/ndx/25520_ivExtractor.ndx'

os.system(cmd_str)