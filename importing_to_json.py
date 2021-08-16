import json
import mido as m
import numpy as np

import importing_midi as midi

# Writing json file as nested list => [[num, result_array.shape[0], 88], [...]]
midi_list = []
for num in range(1, 910):
    midi_file = m.MidiFile(f"POP909-Dataset-master/POP909/{num:03d}/{num:03d}.mid", clip=True)
    result_array = midi.mid2arry(midi_file)
    test = np.array2string(result_array)
    print(f'{num:03d} {result_array.shape[0]}') #testing purposes
    midi_list.append((num, result_array.shape[0], 88))
    with open('midi_json.json', 'w+') as jsonFile:
        json.dump(midi_list, jsonFile)
