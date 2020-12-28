import requests
import lxml.html as html
import time
import os

composer_links = [#'https://www.midiworld.com/bach.htm',
                  #'https://www.midiworld.com/beethoven.htm',
                  #'https://www.midiworld.com/chopin.htm',
                  #'https://www.midiworld.com/mozart.htm',
                  #'https://www.midiworld.com/tchaikovsky.htm',
                  #'https://www.midiworld.com/schumann.htm',
                  #'https://www.midiworld.com/handel.htm',
                  #'https://www.midiworld.com/mendelssohn.htm',
                  ]

for composer_link in composer_links:
  print(f'Going to {composer_link}')
  # Make a request to midiworld, the source of midi files
  response = requests.get(composer_link)
  tree = html.fromstring(response.text)

  # Some debug settings n stuff
  download_midi = True

  # Get all list items
  list_items = tree.xpath('//li')

  # First 6 are header links, so skip that
  for i in range(6, len(list_items)):
    # Get the actual item
    music_li = list_items[i]

    # Seperate the url from raw html stuff
    midi_file_url = str(html.tostring(music_li)).split('"', 2)[1]
    # Split the url by / to get to the filename
    split_url = midi_file_url.split('/')
    # Get the last part of the link, the actual file name
    filename = str(split_url[len(split_url)-1]).split('.')[0]

    # Make sure that the list item is and actual midi file, there are a couple of .htm files for specific composers
    if os.path.splitext(midi_file_url)[1] == '.mid':
      # Make a request to the file url
      midi_file = requests.get(midi_file_url)
      # Check if the file already exists
      if not os.path.exists(f'./data/raw/mozart/{i}_{filename}.mid'):
        # If it doesn't, create a file, dump the midi data in it
        with open(f'./data/raw/mozart/{i}_{filename}.mid', 'wb') as f:
          f.write(midi_file.content)
        print(f"{i}/{len(list_items)}: Grabbed {filename}")
      else:
        # If it exists, skip
        print(f"{filename} already exists")
    else:
      # If its actually a class name or .htm file, skip
      print(f"Could not grab {midi_file_url}")

    # Sleep so I don't get IP banned
    time.sleep(0)