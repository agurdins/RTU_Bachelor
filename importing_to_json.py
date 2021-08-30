import json
import mido as m

import importing_midi as midi

y = []
lengths = []
tempo = []
array_shape = []

for num in range(1, 910):
    midi_file = m.MidiFile(f"POP909-Dataset-master/POP909/{num:03d}/{num:03d}.mid", clip=True)
    result_array = midi.mid2arry(midi_file)
    print(f'{num:03d}') #testing purposes
    BPM = int(m.tempo2bpm(midi.get_tempo(midi_file)))
    y.append(BPM)
    lengths.append(int(midi_file.length))
    tempo.append(int(midi.get_tempo((midi_file))))
    array_shape.append(result_array.shape[0])


# y jau ir BPM
print(len(lengths))

with open('midi_json.json', 'w+') as jsonFile:
    json.dump({
        'shape': (num, max(array_shape), 88),
        'y': y, # BPM scalar vertibas
        'lengths': array_shape,
    }, jsonFile)

with open('midi_json2.json', 'w+') as jsonFile:
    json.dump({
        'shape': (len(lengths), max(lengths), 88), # mmap shape must be like this - Evalds
        'y': y, # BPM scalar vertibas
        'lengths': lengths,
    }, jsonFile)

with open('midi_json3.json', 'w+') as jsonFile:
    json.dump({
        'shape': (len(lengths), max(lengths), max(tempo)),
        'y': y,  # BPM scalar vertibas
        'lengths': lengths,
        'tempo': tempo
    }, jsonFile)