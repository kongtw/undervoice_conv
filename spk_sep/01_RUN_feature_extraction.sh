#!/bin/bash
FEATURE_TYPE="SPro"		# can be SPro or HTK
INPUT_FORMAT="WAV"		# can be SPH or PCM or WAV

# If format is 'WAV', extract MFCC feature directly
if [ $INPUT_FORMAT = "WAV" ]; then
	# Extract MFCC features of register wavs with SPro
	for i in `cat data/rdata.lst`;do
        	COMMAND_LINE="bin/sfbcep -f 8000 -m -k 0.97 -p19 -n 24 -r 22 -e -D -A -F PCM16  data/pcm/spkregister_8k/$i.wav data/prm/$i.tmp.prm"
                echo $COMMAND_LINE
                $COMMAND_LINE
   	done
fi
