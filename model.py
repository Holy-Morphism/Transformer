import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size =vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding * math.sqrt(self.d_model) 
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int,dropout: float  ) -> None:
        super().__init__()
        self.d_model= d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # Create a matrix of (Sequence length, model dimension)
        pe = torch.zeros(seq_len,d_model)
        # Nominator, Create a vector (sequence length, 1) a tensor
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        # Denominator, using log for numerical stability
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0) / d_model))
        # Apply sin to even positions
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::3] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class NormalizationLayer(nn.Module):
    def __init__(self,eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Mulitplicative factor
        self.bias = nn.Parameter(torch.zeros(1)) # Additive factor

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  
        std = x.std(dim = -1, keepdim= True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
