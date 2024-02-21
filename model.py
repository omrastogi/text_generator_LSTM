import os
import torch
import torch.nn as nn
import string
import random
import numpy as np
import sys
import unidecode
from torch.utils.tensorboard import SummaryWriter

# Ensure 'tensorboard' is installed. If not, run: pip install tensorboard

all_characters = string.printable
n_characters = len(string.printable)


def get_names_from_file(file_path):
    names = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Iterate over lines, skipping the header line
    for line in lines[1:]:
        name = line.strip().split(',')[1]
        names.append(name)
    return names

# file = unidecode.unidecode(open('names.txt').read())
file = get_names_from_file('names.txt')
file = " ".join(file)
# Make sure to set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class Generator:
    def __init__(self, model=None):
        self.chunk_len = 250
        self.num_epochs = 300
        self.batch_size = 1
        self.print_every = 10
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.03
        self.input_size = self.output_size = n_characters
        self.checkpoint_interval = 50
        if model:
            self.rnn = model

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])
        return text_input.long(), text_target.long()

    def generate(self, initial_str='Ab', predict_len=100, temperature=0.85, how_many=1, max_length=float('inf')):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)

        all_pred = []
        all_probs = []  # List to store probabilities for each generated name
        last_char = initial_input[-1]

        for i in range(how_many):
            prob_sequence = []  # List to store probabilities for the current generated name
            for p in range(predict_len):
                output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
                output_dist = output.data.view(-1).div(temperature).exp()
                top_char = torch.multinomial(output_dist, 1)[0]
                predicted_char = all_characters[top_char]

                if predicted_char in ["\n", " "]:
                    break

                prob_sequence.append(output_dist[top_char].item())  # Append probability
                predicted += predicted_char
                last_char = self.char_tensor(predicted_char)

            if len(predicted) < max_length:
                all_pred.append(predicted)
                all_pred.append(prob_sequence)
                all_probs.append(prob_sequence)

        return all_pred, all_probs

    """def generate(self, initial_str='Ab', predict_len=100, temperature=0.85, how_many=1, max_length=float('inf')):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)

        all_pred = []
        last_char = initial_input[-1]
        # Also require a total loss for the name
        for i in range(how_many):
        prob=[]
            for p in range(predict_len):
                output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
                output_dist =( output.data.view(-1).div(temperature).exp())
                top_char = torch.multinomial(output_dist, 1)[0]
                predicted_char = all_characters[top_char]
                char_prob=output_dist[top_char]
                prob.append(char_prob)

                if predicted_char in ["\n", " "]:
                    break
                predicted += predicted_char
                last_char = self.char_tensor(predicted_char)
            if len(predicted) < max_length:
                all_pred.append(predicted)
        return [all_pred,prob]                                                                                                                                                                
    """

    '''def generate(self, initial_str='Ab', predict_len=100, temperature=0.85, how_many=1, max_length=float('inf')):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)

        all_pred = []
        last_char = initial_input[-1]

        for i in range(how_many):
            for p in range(predict_len):
                output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
                output_dist = output.data.view(-1).div(temperature).exp().cpu().numpy()

                # Sample characters with higher probabilities
                top_n = 5  # Choose top 5 characters
                top_indices = output_dist.argsort()[-top_n:][::-1]
                top_probabilities = output_dist[top_indices]
                top_probabilities = top_probabilities / np.sum(top_probabilities)  # Normalize probabilities
                sampled_index = np.random.choice(top_indices, p=top_probabilities)

                predicted_char = all_characters[sampled_index]

                if predicted_char in ["\n", " "] or len(predicted) >= max_length:
                    break
                predicted += predicted_char
                last_char = self.char_tensor(predicted_char)

            if len(predicted) < max_length:
                all_pred.append(predicted)

        return all_pred'''

    def train(self):
        self.rnn = RNN(n_characters, self.hidden_size, self.num_layers, n_characters).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print("=> Starting training")
        min_loss = float('inf')
        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}], Loss: {loss}')
                # print(self.generate())

            if loss < min_loss:
                min_loss = loss
                self.save_checkpoint(epoch, self.rnn, optimizer, loss)

        self.save_checkpoint(epoch, self.rnn, optimizer, loss)

    def save_checkpoint(self, epoch, model, optimizer, loss):
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}_{loss}.pth'
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }
        torch.save(state, checkpoint_path)


if __name__ == "__main__":
# Instantiate the Generator and call the train method
    generator = Generator()
    generator.train()