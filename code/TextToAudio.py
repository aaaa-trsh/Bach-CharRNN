import pretty_midi
import numpy as np
import sys
import scipy.signal
import scipy.io.wavfile
import math
import random

np.set_printoptions(threshold=np.inf)
def generate_audio(name):
  print(f'Loading {name}')
  # Load data from dir
  text_name = name
  text_dir = f'./generated/{text_name}.txt'
  print(f'Reading {name}')
  print(type(text_dir))
  text = open(text_dir, 'rb').read().decode(encoding='utf-8')
  print(f'Read {name}')

  # Convert piano roll to midi object from examples
  def piano_roll_to_pretty_midi(piano_roll, fs=32, program=0):
    print('Converting piano roll into midi')
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
      # use time + 1 because of padding above
      velocity = piano_roll[note, time + 1]
      time = time / fs
      if velocity > 0:
        if prev_velocities[note] == 0:
          note_on_time[note] = time
          prev_velocities[note] = velocity
      else:
        pm_note = pretty_midi.Note(
          velocity=prev_velocities[note],
          pitch=note,
          start=note_on_time[note],
          end=time)
        instrument.notes.append(pm_note)
        prev_velocities[note] = 0
    pm.instruments.append(instrument)
    print('Done')
    return pm

  # For every line in the text file:
  #  -Get each playing note
  #  -Turn it on at timestep {line}
  text_lines = text.split(' ')
  piano_roll = np.zeros([128, len(text_lines)])
  notes = 0
  print(f'Decoding {len(text_lines)} timesteps into notes')

  for line in range(len(text_lines)):
    note_ids = text_lines[line].strip()#[1::2]
    if len(note_ids) != 0:
      note_ids = ''.join(random.sample(note_ids,len(note_ids)))
      for char in note_ids:
        timestep = line
        try:
          note_id = ord(char) + 20
          piano_roll[note_id][timestep] = 127
        except:
          print('', end='')
        #print(note_id + 32, timestep, end=' ')
  print('Finished decoding text')
  midi = piano_roll_to_pretty_midi(piano_roll)
  print(f'Writing to ./generated/{text_name}.mid')
  midi.write(f'./generated/{text_name}.mid')
  print('Done')
  print(f'Creating audio from MIDI')
  synthesized = midi.synthesize(20000, np.sin)
  print('Done')
  print('Saving WAV to file')
  scipy.io.wavfile.write(f'./generated/{text_name}.wav', 20000, synthesized)
  print(f'Finished exporting {text_name}')