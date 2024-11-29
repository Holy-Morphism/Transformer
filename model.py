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

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model: int,d_ff :int,dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model,d_ff) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.linear_1(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttenion(nn.module):

    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 # model dimension should be divisible by number of heads
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) #WQ
        self.w_k = nn.Linear(d_model, d_model) #WK
        self.w_v = nn.Linear(d_model, d_model) #WV
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #  (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value) , attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_model) --> (batch, h, seq_len, d_model)
        query = query.view(query.shape[0],query.shape[1], self.h , self.d_model).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h , self.d_model).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h , self.d_model).transpose(1,2)

        x, self.attention_scores = MultiHeadAttenion.attention(query,key,value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],self.h * self.d_k )

        return self.w_o(x)






