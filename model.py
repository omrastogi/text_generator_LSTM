import os
import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
from torch.utils.tensorboard import SummaryWriter

# Ensure 'tensorboard' is installed. If not, run: pip install tensorboard

all_characters = string.printable
n_characters = len(string.printable)

file = unidecode.unidecode(open('names.txt').read())

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
        self.num_epochs = 500
        self.batch_size = 1
        self.print_every = 10
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.03
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
        last_char = initial_input[-1]
        for i in range(how_many):
            for p in range(predict_len):
                output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
                output_dist = output.data.view(-1).div(temperature).exp()
                top_char = torch.multinomial(output_dist, 1)[0]
                predicted_char = all_characters[top_char]
                if predicted_char == "\n":
                    break
                predicted += predicted_char
                last_char = self.char_tensor(predicted_char)
            if len(predicted) < max_length:
                all_pred.append(predicted)

        return all_pred

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
        }
        torch.save(state, checkpoint_path)


if __name__ == "__main__":
# Instantiate the Generator and call the train method
    generator = Generator()
    generator.train()