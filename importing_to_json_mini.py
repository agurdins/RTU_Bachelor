import json
import mido as m

import importing_midi as midi

y = []
lengths = []

for num in range(1, 10):
    midi_file = m.MidiFile(f"POP909-Dataset-master/POP909/{num:03d}/{num:03d}.mid", clip=True)
    result_array = midi.mid2arry(midi_file)
    print(f'{num:03d}') #testing purposes
    BPM = int(m.tempo2bpm(midi.get_tempo(midi_file)))
    y.append(BPM)
    if result_array.shape[0] > 15000:
        lengths.append(15000)
    else:
        lengths.append(result_array.shape[0])

with open('midi_json_mini.json', 'w+') as jsonFile:
    json.dump({
        'shape': (len(lengths), max(lengths), 88), # mmap shape must be like this - Evalds
        'y': y, # BPM scalar vertibas
        'lengths': lengths,
    }, jsonFile)
