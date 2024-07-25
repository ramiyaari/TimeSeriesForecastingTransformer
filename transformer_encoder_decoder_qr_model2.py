import torch
import torch.nn as nn
import math
import random


class TransformerEncoderDecoderQRModel2(nn.Module):
    def __init__(self, nhead, d_model, d_hid, nlayers, 
                 input_len, target_len, nfeatures, nquantiles, dropout=0.5):
        super(TransformerEncoderDecoderQRModel2, self).__init__()
        self.d_model = d_model
        self.model_type = 'Transformer'
        self.input_len = input_len
        self.target_len = target_len
        self.nfeatures = nfeatures
        self.nquantiles = nquantiles

        self.pos_encoder = PositionalEncoding(d_model, max(input_len, target_len), dropout)
        self.src_linear = nn.Linear(nfeatures, d_model)
        self.tgt_linear = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(d_model, nhead, nlayers, nlayers, d_hid, dropout)
        self.decoder = nn.Linear(d_model, nquantiles)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def init_weights(self):
        initrange = 0.1
        self.src_linear.bias.data.zero_()
        self.src_linear.weight.data.uniform_(-initrange, initrange)
        self.tgt_linear.bias.data.zero_()
        self.tgt_linear.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        
        # tgt_seq_len = tgt.shape[1]
        # if self.tgt_mask is None or self.tgt_mask.size(0) != tgt_seq_len:
        #     self.tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        if(src.dim() == 2):
            src = src.unsqueeze(-1) # [B, seq_len] -> [B, seq_len, 1]
        if(tgt is not None and tgt.dim() == 2):
            tgt = tgt.unsqueeze(-1) # [B, seq_len] -> [B, seq_len, 1]

        src = self.src_linear(src)
        src = self.pos_encoder(src)

        if(tgt is not None):
            # Training mode: Use teacher forcing
            tgt_seq_len = tgt.shape[1]
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
            tgt = self.tgt_linear(tgt)
            tgt = self.pos_encoder(tgt)
            # Forward pass through the transformer
            # Note: Transformer expects src and tgt in shape [seq_len, batch_size, feature_size]
            output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1), tgt_mask=tgt_mask).transpose(0, 1)
            output = self.decoder(output)
            # outputs0 = []
            # outputs = []
            # for t in range(tgt_seq_len):
            #     if t == 0 or random.random() < teacher_forcing_ratio:
            #         tgt_input = tgt[:, :t+1, :]
            #     else:
            #         tgt_input = torch.cat([tgt[:, :t, :], outputs0[-1]], dim=1)
            #     output0 = self.transformer(src.transpose(0, 1), tgt_input.transpose(0, 1), tgt_mask=tgt_mask[:t+1, :t+1]).transpose(0, 1)
            #     output0 = output0[:, -1, :]
            #     outputs0.append(output0.unsqueeze(1))
            #     output = self.decoder(output0)
            #     outputs.append(output.unsqueeze(1))
            # output = torch.cat(outputs, dim=1)
        else:
            # Inference mode: Autoregressive decoding
            batch_size = src.shape[0]
            output_seq = []
            SOS = torch.tensor(-2.0, dtype=torch.float32)
            tgt_input = torch.full((batch_size, 1, 1), SOS).to(src.device)  # Start with SOS token
            for _ in range(self.target_len):
                tgt = self.tgt_linear(tgt_input[:, -1:, :])
                tgt = self.pos_encoder(tgt)
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
                output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1), tgt_mask=tgt_mask).transpose(0, 1)
                output = self.decoder(output[:, -1, :])  # Take the last output
                output_seq.append(output.unsqueeze(1))
                # Select the median quantile (assuming it's the middle one if nquantiles is odd)
                median_index = self.nquantiles // 2
                median_output = output[:, median_index].unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1]
                tgt_input = torch.cat((tgt_input, median_output), dim=1)
            output = torch.cat(output_seq, dim=1)

        return output 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model), adding batch dimension as 1 for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]  # pe is automatically broadcasted to the batch size
        return self.dropout(x)

