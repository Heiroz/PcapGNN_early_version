import torch
import torch.nn as nn
import math

class Generator(nn.Module):
    def __init__(self, noisy_size, output_size, condition_size, hidden_size=512 * 6, 
                 num_layers=3, num_heads=2, dropout=0.1):
        super(Generator, self).__init__()
        self.noisy_size = noisy_size
        self.output_size = output_size
        self.condition_size = condition_size
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=noisy_size + condition_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.positional_encoding = PositionalEncoding(noisy_size + condition_size, dropout)
        self.fc = nn.Linear(noisy_size + condition_size, output_size)

    def forward(self, noisy, condition_vector):
        batch_size = noisy.size(0)
        src = torch.cat((noisy, condition_vector), dim=1)
        src = self.positional_encoding(src.unsqueeze(1))
        encoded = self.transformer_encoder(src)
        
        generated_samples = self.fc(encoded.squeeze(1))
        return generated_samples.view(batch_size, -1, self.output_size)


class Generator_sigmoid(nn.Module):
    def __init__(self, noisy_size, output_size, condition_size, 
                 hidden_size=512, num_layers=4, num_heads=2, dropout=0.1):
        super(Generator_sigmoid, self).__init__()
        self.noisy_size = noisy_size
        self.output_size = output_size
        self.condition_size = condition_size

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=noisy_size + condition_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.positional_encoding = PositionalEncoding(noisy_size + condition_size, dropout)
        self.fc = nn.Linear(noisy_size + condition_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, noisy, condition_vector):
        batch_size = noisy.size(0)
        src = torch.cat((noisy, condition_vector), dim=1)
        src = self.positional_encoding(src.unsqueeze(1))
        encoded = self.transformer_encoder(src)
        generated_samples = self.sigmoid(self.fc(encoded.squeeze(1)))
        binary_generated_samples = (generated_samples > 0.6).float()
        return binary_generated_samples.view(batch_size, -1, self.output_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

