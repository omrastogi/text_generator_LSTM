import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # repeat decoder hidden state max_len times
        hidden = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]

        # calculate attention scores
        attn_energies = self.score(hidden, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.attention = Attention(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        attention_weights = self.attention(output[-1], hidden[0])
        context = attention_weights.bmm(output.transpose(0, 1))
        output = output.squeeze(0)
        context = context.squeeze(1)
        output = self.out(torch.cat((output, context), 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attention_weights

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))


    def train(self):
        self.lstm_with_attention = LSTMWithAttention(n_characters, self.hidden_size, n_characters, self.num_layers).to(
            device)

        optimizer = torch.optim.Adam(self.lstm_with_attention.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print("=> Starting training")
        min_loss = float('inf')
        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden = self.lstm_with_attention.init_hidden()

            self.lstm_with_attention.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, hidden, _ = self.lstm_with_attention(inp[:, c], hidden)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}], Loss: {loss}')
                # print(self.generate())

            if loss < min_loss:
                min_loss = loss
                self.save_checkpoint(epoch, self.lstm_with_attention, optimizer, loss)

        self.save_checkpoint(epoch, self.lstm_with_attention, optimizer, loss)

# Example usage:
input_size = 128  # Assuming input vocabulary size
hidden_size = 256
output_size = 128  # Same as input size for autoencoder-like generation
num_layers = 2

model = LSTMWithAttention(input_size, hidden_size, output_size, num_layers)
print(model)
