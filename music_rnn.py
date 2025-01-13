import collections
import datetime 
import glob
from typing import Optional
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import fluidsynth
import pretty_midi
from IPython import display
import tensorflow as tf


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000

# Downloads the maestro dataset, containing around 1200 midi files.
data_dir = pathlib.Path('data/maestro-v2_extracted/maestro-v2.0.0')
if not data_dir.exists():
    
  tf.keras.utils.get_file(
      'maestro-v2.0.0-midi.zip',
      origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
      extract=True,
      cache_dir='.', cache_subdir='data',
  )

filenames = glob.glob(str(data_dir/'**/*.midi'))
#print('Number of files:', len(filenames))
# At this stage, output should be "Number of files: 1282"

# Now, practice using pretty_midi to parse a single MIDI file and inspect format of the notes
sample_file = filenames[0]

pm = pretty_midi.PrettyMIDI(sample_file)

def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    waveform = pm.fluidsynth(fs = _SAMPLING_RATE)

    waveform_short = waveform[ :seconds*_SAMPLING_RATE]
    return display.Audio(waveform_short, rate = _SAMPLING_RATE)

display_audio(pm)