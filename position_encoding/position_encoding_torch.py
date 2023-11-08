class PositionEncoding(torch.nn.Module):
  def __init__(self, hidden_size,
         dropout = 0.1, maxlen = 10, return_pos_embedding = False, device = None):
    super().__init__()
    den = torch.exp(- torch.arange(0, hidden_size, 2) * math.log(10000)/ hidden_size)
    pos = torch.arange(0, maxlen).reshape(maxlen, 1)
    
    pos_embedding = torch.zeros((maxlen, hidden_size))
    
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    
    pos_embedding = pos_embedding.unsqueeze(0)
    if device:
      pos_embedding = pos_embedding.to(device)

    self.return_pos_embedding = return_pos_embedding
    
    self.dropout = torch.nn.Dropout(dropout)
    self.register_buffer('pos_embedding', pos_embedding)
        
    self.device = device
    
  def forward(self, token_embedding):
    pos = self.pos_embedding[:, :token_embedding.size(1), :]
    if self.return_pos_embedding:
      return self.dropout(token_embedding + pos), pos
    return self.dropout(token_embedding + pos)