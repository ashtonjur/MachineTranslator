class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads)
        self.ffn = PositionwiseFeedForward(embed_size, ff_hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        attn_out, _ = self.attn(src, src, src, mask)
        src = self.norm1(src + self.dropout(attn_out))

        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_out))

        return src


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(embed_size, num_heads)
        self.attn2 = MultiHeadAttention(embed_size, num_heads)
        self.ffn = PositionwiseFeedForward(embed_size, ff_hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        attn_out1, _ = self.attn1(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out1))

        attn_out2, _ = self.attn2(tgt, src, src, src_mask)
        tgt = self.norm2(tgt + self.dropout(attn_out2))

        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))

        return tgt

