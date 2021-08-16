import json
import mido as m
import importing_midi as midi

# Writing json file as nested list
metaData = []
for num in range(1, 910):
    midiFile = m.MidiFile(f"POP909-Dataset-master/POP909/{num:03d}/{num:03d}.mid", clip=True)
    result_array = midi.mid2arry(midiFile)
    BPM = int(m.tempo2bpm(midi.get_tempo(midiFile)))
    print(f'{num:03d} {BPM}')  # testing purposes
    metaData.append((num, result_array.shape, BPM))
    with open('metadata2.json', 'w+') as jsonFile:
        json.dump(metaData, jsonFile)
