import os
import unidecode
import string
import random
import re
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-s', '--save_path', help='Relative file location to the save path')
parser.add_argument('-l', '--load_path', help='Relative file location to the load path')
parser.add_argument('-d', '--data_path', help='Relative directory location to the data set with the input.txt file')

args = parser.parse_args()

PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = None
LOAD_PATH = None
DATA_PATH = None
if args.save_path:
    SAVE_PATH = os.path.join(PATH, args.save_path)
    print("LOAD_PATH", SAVE_PATH)
if args.load_path:
    LOAD_PATH = os.path.join(PATH, args.load_path)
    print("LOAD_PATH", LOAD_PATH)
if args.data_path:
    DATA_PATH = os.path.join(PATH, args.data_path)
    print("DATA_PATH", DATA_PATH)

all_characters = string.printable
n_characters = len(all_characters)

if DATA_PATH:
    file = unidecode.unidecode(open(os.path.join(PATH, DATA_PATH, "input.txt")).read())    
else:
    file = unidecode.unidecode(open(os.path.join(PATH, "data", "reinfocementTextbook", "input.txt")).read())
file_len = len(file)
# print('file_len =', file_len)

chunk_len = 200

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

# print(random_chunk())


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size)).type(torch.cuda.FloatTensor)


# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        if string[c] in all_characters:
            tensor[c] = all_characters.index(string[c])
    return Variable(tensor).type(torch.cuda.LongTensor)

# print(char_tensor('abcDEF'))

def random_training_set():    
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / chunk_len


n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

if LOAD_PATH != None:
    decoder.load_state_dict(torch.load(LOAD_PATH))

for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set())
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('Wh', 100), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

if SAVE_PATH != None:
    torch.save(decoder.state_dict(), SAVE_PATH)