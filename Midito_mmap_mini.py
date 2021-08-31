import numpy as np
import mido as m
import importing_midi as midi
import json

with open('midi_json_mini.json', 'r') as midiJ:
    json_data = json.load(midiJ)

shape = tuple(json_data['shape'])
length = json_data['lengths']
midi_memmap = "POP909-Dataset-master/POP909/memmap_mini.dat"
fpath = np.memmap(midi_memmap, dtype='float32', mode='w+', shape=shape)

for num in range(1, len(json_data['y'])):
    midi_file = m.MidiFile(f"POP909-Dataset-master/POP909/{num:03d}/{num:03d}.mid", clip=True)
    result_array = midi.mid2arry(midi_file)
    if length == 15000:
        fpath[num, :15000, :] = result_array[:15000]
    else:
        for i in length:
            fpath[num, :i, :] = result_array[:15000]
    print(f'{num:03d} {result_array.shape}')

