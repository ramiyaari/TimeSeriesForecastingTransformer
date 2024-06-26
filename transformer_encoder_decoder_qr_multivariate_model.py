import torch
import torch.nn as nn
import math


class TransformerEncoderDecoderQRMultiVariateModel(nn.Module):
    def __init__(self, nhead, d_model, d_hid, nlayers, 
                 input_len, target_len, input_dim, target_dim,
                 nquantiles, dropout=0.5):
        super(TransformerEncoderDecoderQRMultiVariateModel, self).__init__()
        self.d_model = d_model
        self.input_len = input_len
        self.target_len = target_len
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.nquantiles = nquantiles
        self.model_type = 'Transformer'
        self.tgt_mask = None
        max_len = max(input_len,target_len)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.src_linear = nn.Linear(input_dim, d_model)
        self.tgt_linear = nn.Linear(target_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, nlayers, nlayers, d_hid, dropout)
        self.decoder = nn.Linear(d_model, nquantiles*target_dim)
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

    def forward(self, src, tgt):
        
        tgt_seq_len = tgt.shape[1]
        if self.tgt_mask is None or self.tgt_mask.size(0) != tgt_seq_len:
            self.tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        if(src.dim()==2):
            src = src.unsqueeze(-1) # [B, seq_len] -> [B, seq_len, 1]
        if(tgt.dim()==2):
            tgt = tgt.unsqueeze(-1) # [B, seq_len] -> [B, seq_len, 1]

        src = self.src_linear(src)
        src = self.pos_encoder(src)
        tgt = self.tgt_linear(tgt)
        tgt = self.pos_encoder(tgt)

        # Forward pass through the transformer
        # Note: Transformer expects src and tgt in shape [seq_len, batch_size, feature_size]
        output  = self.transformer(src.transpose(0, 1),
                                   tgt.transpose(0, 1),
                                   tgt_mask=self.tgt_mask).transpose(0, 1)

        output = self.decoder(output)
        output = output.view(-1, self.target_len, self.target_dim, self.nquantiles).permute(0, 1, 3, 2)  # Reshape and permute to [batch_size, nquantiles, target_dim]
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

