
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from torch.hub import download_url_to_file
import torch.utils.data

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4

RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 2
RNN_DROPOUT = 0.3

run_path = ''

DEVICE = 'cpu'
# if torch.cuda.is_available():
#     DEVICE = 'cuda'

MIN_SENTENCE_LEN = 3
MAX_SENTENCE_LEN = 20
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
# if DEVICE == 'cuda':
#     MAX_LEN = None

PATH_DATA = '../data'
os.makedirs('./results', exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)
PATH_DATASET = './'

class DatasetCustom(torch.utils.data.Dataset):
    def __init__(self):
        with open(f'{PATH_DATASET}/midi_json.json') as fp:
            self.metadata = json.load(fp)
        shape = tuple(self.metadata['shape'])
        self.mmap = np.memmap('POP909-Dataset-master/POP909/memmap.dat', mode='r', shape=shape)

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.mmap[idx])
        y = torch.FloatTensor(self.metadata['y'][idx])
        length = torch.LongTensor(self.metadata['lengths'][idx])
        return x, y, length


torch.manual_seed(0)
dataset_full = DatasetCustom()
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, lengths=[int(len(dataset_full)*0.8), len(dataset_full)-int(len(dataset_full)*0.8)])
torch.seed()

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = dataset_full.metadata['shape'][-1] # 88

        self.fc_first = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, RNN_HIDDEN_SIZE),
            torch.nn.BatchNorm1d(self.input_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE),
        )

        self.rnn = torch.nn.LSTM(
            input_size=RNN_HIDDEN_SIZE,
            hidden_size=RNN_HIDDEN_SIZE,
            batch_first=True,
        )

        self.fc_last = torch.nn.Sequential(
            torch.nn.Linear(RNN_HIDDEN_SIZE, out_features=RNN_HIDDEN_SIZE),
            torch.nn.BatchNorm1d(RNN_HIDDEN_SIZE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(RNN_HIDDEN_SIZE, out_features=1)
        )

    def forward(self, x: PackedSequence, lengths):

        x_proj = PackedSequence(
            data=self.fc_first.forward(x.data.argmax(dim=1)),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        x_rnn, _ = self.rnn.forward(x_proj)

        # x_rnn.shape = (B, seq, hidden)
        x_rnn_padded = pad_packed_sequence(x_rnn, batch_first=True)
        # x_rnn_padded.shape = (B, max_seq, hidden) Tensor

        x_temp_pooling = []
        for idx in range(x_rnn_padded.size(0)):
            length = lengths[idx]
            x_rnn_padded_each = x_rnn_padded[idx]
            x_rnn_padded_cut = x_rnn_padded_each[:length] # (seq, hidden)
            # var izmantot ari max, last
            x_rnn_padded_avg = torch.mean(x_rnn_padded_cut, dim=0) # hidden
            x_temp_pooling.append(x_rnn_padded_avg)

        x_temp_pooling_tensor = torch.stack(x_temp_pooling) # (B, hidden)
        y_prim = self.fc_last.forward(x_temp_pooling_tensor)

        y_prim = torch.relu(y_prim) # jo nevar bpm nevar but negativs
        y_prim = y_prim.squeeze() # (B, 1) => (B, )

        return y_prim

model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

print(DEVICE)

metrics = {}
best_test_loss = float('Inf')
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS+1):


    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y, lengths in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            lengths = lengths.to(DEVICE)

            x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

            y_prim = model.forward(x_packed)

            loss = torch.mean(torch.abs(y - y_prim))

            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if best_test_loss > loss.item():
        best_test_loss = loss.item()
        torch.save(model.cpu().state_dict(), f'./results/model-{epoch}.pt')
        model = model.to(DEVICE)


    plt.figure(figsize=(12,5))
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.savefig(f'./results/epoch-{epoch}.png')
    plt.show()