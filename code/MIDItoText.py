import pretty_midi
import py_midicsv
import progressbar
import glob
import os
import random
from subprocess import call 
from datetime import datetime

# If translating midi to text
translate = True

# If compiling data into a text file
compile = True

def get_all_files(dir, ext, seed_int=1):
  files = glob.glob(f'{dir}*{ext}')
  #random.seed(datetime.now())
  #random.shuffle(files)
  return files

midi_files = get_all_files(r'./data/raw/kirby/', '.mid', 4)
file_index = 0
print(len(midi_files))
if translate:
  for file in midi_files:
    try:
      samples = 32
      cur_midi = pretty_midi.PrettyMIDI(file)
      notes = []
      piano_roll = cur_midi.get_piano_roll(samples)[20:108]
      filename = os.path.split(os.path.splitext(file)[0])[1]
      print(f"\n\nTranslating {filename.upper()}\n-----------------")
      bar = progressbar.ProgressBar(max_value=int(round((cur_midi.get_end_time()/2) * 100)), widgets=[f'{file_index}/{len(midi_files)} Translating {filename}: ',progressbar.Bar(), progressbar.ETA()])

      # For each timestep, log
      info = ''
      for timestep in range(int(round((cur_midi.get_end_time()/2) * 100))):
        notes_on = []
        note_chars = []

        for note_id in range(88):
          try:
            if piano_roll[note_id + 33][timestep-1] > 64:
              notes_on.append(pretty_midi.note_number_to_name(note_id + 33))
              note_chars.append(chr(note_id + 33))
          except:
            print('', end="")
        if len(notes_on) == 0:
          notes_on.append(" ")

        info = info + ''.join(note_chars) + ' '
        bar.update(timestep)
      with open(f"./data/text/kirby/{filename}.txt", "w") as file:
        file.seek(0)
        file.truncate()
        file.write(str(info))
      file_index += 1
    except:
      print("Whoops! Something went wrong")

if compile:
  text_files = get_all_files(r'./data/text/kirby/', '.txt', random.randint(0, 250))#[:250]

  with open(f"./data/kirby_data.txt", "w") as file:
    file.seek(0)
    file.truncate()
  for file_path in text_files:
    #if os.path.getsize('./data/data.txt') > 2500000:
    #  break
    print(file_path)
    with open(file_path, 'r') as file:
      lines = file.readlines()
    with open(f"./data/kirby_data.txt", "a") as file:
      file.writelines(lines)
