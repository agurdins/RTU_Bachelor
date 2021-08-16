import numpy as np
import mido as m
import importing_midi as midi
import json

with open('midi_shapes.json', 'r') as midiJ:
    json_data = json.load(midiJ)


shape = json_data['shape']
midi_memmap = "POP909-Dataset-master/POP909/memmap.dat"
fpath = np.memmap(midi_memmap, dtype='float32', mode='w+', shape=json_data['shape'])

for num in range(1, 910):
    midi_file = m.MidiFile(f"POP909-Dataset-master/POP909/{num:03d}/{num:03d}.mid", clip=True)
    result_array = midi.mid2arry(midi_file)
    test = np.array2string(result_array)
    print(f'{num:03d} {result_array.shape}')

    fpath[num, :result_array.shape[0], :] = result_array[:]