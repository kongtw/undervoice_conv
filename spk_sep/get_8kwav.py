import commands
import glob

for wav_name in glob.glob('./data/pcm/spkregister/*'):
    dest_name = './data/pcm/spkregister_8k/' + wav_name.split('/')[-1]
    print dest_name
    sox_cmd = 'sox -r 6k ' + wav_name + ' -r 8k ' + dest_name
    (status, output) = commands.getstatusoutput(sox_cmd)
