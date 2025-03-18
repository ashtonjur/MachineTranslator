class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_heads, num_layers, ff_hidden_size, dropout=0.1):
        super(Transformer, self).__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)

        self.positional_encoding = PositionalEncoding(embed_size)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)])

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_embed = self.src_embedding(src) + self.positional_encoding(src)
        tgt_embed = self.tgt_embedding(tgt) + self.positional_encoding(tgt)

        for layer in self.encoder_layers:
            src_embed = layer(src_embed, src_mask)

        for layer in self.decoder_layers:
            tgt_embed = layer(tgt_embed, src_embed, tgt_mask, src_mask)

        output = self.fc_out(tgt_embed)

        return output
