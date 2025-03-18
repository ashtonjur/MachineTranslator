# Transformer-based Machine Translation

This repository contains a PyTorch implementation of a Transformer model for machine translation. The model is trained to translate sentences from English to Polish using the OpenSubtitles dataset.
The transformer encoder-decoder architecture is designed to handle tasks that require
taking input in one language and returning in another. The encoder takes the input
sentence and creates a fixed-size vector representation of it, which is then given to the
decoder to generate the output sentence. The decoder utilizes both self-attention and
cross-attention, where the attention mechanism is applied to the output of the encoder
and the input of the decoder.

## Requirements

- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/transformer-translation.git
    cd transformer-translation
    ```

2. Install the dependencies:
    ```sh
    pip install torch
    ```

3. Download the dataset (OpenSubtitles) and place the files `OpenSubtitles.en-pl.en` and `OpenSubtitles.en-pl.pl.txt` in the repository directory.

## Model Architecture

The Transformer model consists of:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Position-wise Feed-Forward Networks
- Positional Encoding

The encoder and decoder layers are composed of these modules.

## Position-wise Feed-Forward

To implement the encoder-decoder architecture, it is necessary to create a Positionwise
Feed-Forward Network (FFN). This module allows processing the representation of
each token and enhances the model’s ability to learn more advanced patterns. The
FFN consists of two layers, or a multi-layer perceptron (MLP). Both layers, fully
connected and separated by a ReLU activation, add non-linearity to capture intricate
relationships.
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

```
The first transformation layer (fc1) maps the input from its original embedding size to
a higher-dimensional hidden space. This enables the model to learn richer and more
complex feature representations of the input data by projecting it into a space with
more dimensions. Then the ReLU function introduces nonlinearity into the system,
ensuring that only positive values are propagated. The second layer (fc2) projects
the hidden representation back to the original embedding size. Thanks to that, the
high-dimensional features are introduced to the hidden space in the representation that
aligns with the model’s architecture. One can test the FFN by giving it some simple
tokens such as "The", "cat", "sat" and then giving them example token IDs.

## Positional Encoding
Unlike RNNs or LSTMs, transformers do not have an inherent mechanism to process
input sequentially. That is because of the self-attention, which treats the input as a
set of independent tokens. To inject information about the order of tokens into their
embeddings, Positional Encoding is being used. Thanks to that, the model is able to
recognize sequence’s structure.
```python

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp
        (torch.arange(
        0, embed_size, 2
         ).float() * -(math.log(10000.0) / embed_size)
         )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)


    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :].to(x.device)

```
This module generates a fixed set of sinusoidal patterns for each position and dimension
of the embedding. The encoding uses a combination of sine and cosine functions,
with frequencies determined by the position and embedding index. Specifically, sine
functions are applied to even indices, while cosine functions are used for odd indices.
The result is a set of unique and continuous positional values that provide a smooth
gradient across the embedding space, making them easy for the model to interpret and
learn from.

## Scaled-dot Product Attention

To calculate how much emphasis to put on other parts of the input sentence, each
input word must get its own score in relation to the word being considered. The score
is being calculated by multiplying the dot product of the query vector with the key
vector of the respective word we’re scoring.

To achieve this, it is necessary to use the Scaled Dot-Product Attention mechanism,
which scales down the dot products.

```python

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(
        query, key.transpose(-2, -1)) / math.sqrt(
        query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

```
Thanks to Scaled-Dot Product Attention mechanism, the dot product is is computed
between query and the transpose of key along the last two dimensions. This results in a score matrix of shape (batch_size, num_heads, seq_len, seq_len). Each entry
in the score matrix indicates how much a position in the query sequence attends to a
position in the key sequence. Then, the mechanism applies the mask if it’s provided,
making sure that it’s ignored if its equal to 0. Finally it applies the softmax function
along the last dimension (dim=-1) of the scores and Performs a weighted sum of the
value vectors using the attention weights

## Multi-Head Attention
The multiplication of the Softmax and the value vector yields a sum of the weighted
value vectors. This way, the output of the self attention layer at this position (for the
respective word) is produced. With the multi-headed attention there are multiple sets
of Query-Key-Value weight matrices – each one randomly initialized. After training

each set is used to project the input embeddings into a different representation sub-
space. Because of that, the model ends up with the same amount of matrices as the

amount of heads when the feed-forward mechanism is only expecting one matrix. In
order to solve the problem, all the matrices are being concat to then be multiplied by
an additional weight matrix WO – a matrix trained jointly with the model.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.attn = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        N = query.size(0)
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        Q = Q.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        out, attn_weights = self.attn(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(N, -1,
        self.num_heads * self.head_dim)
        out = self.fc_out(out)
        return out, attn_weights
```
## Encoder and Decoder
All of those mechanisms allow us to eventually create the Encoder Layer. It takes in an
input sequence and processes it to create a set of context vectors. Encoder converts the
text into a numerical value and feeds with it the layers that reduce their dimensionality
while preserving relevant information about how they relate to one another within the
sentence structure.

```python

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

```
The Encoder Layer calls the Multi-Head Attention and normalizes the output added
to the original output. Then it passes the normalized tensor through the Positionwise

Feed-Forward Network to apply the two linear transformations with the ReLU activation in between. The output is added to the post-attention input and normalizes the
final output once again.
This encoded version of each sentence is then sent to a Decoder Layer for further
processing. The Decoder’s role is to take Encoder’s representation and reconstruct it
back into its original form. Thanks to the attention mechanism it can establish the
link between what was encoded and what is supposed to be decoded.

```python

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

```
The Decoder Layer uses the Multi-Head Attention mechanism to compute self-attention
over a target sequence and, analogously to the Encoder Layer, normalizes the sum of

the original target value and the output of the self-attention module. The second self-
attention layer then calculates the cross-attention between the target sequence and the

source sequence. It updates the target to then pass it through the Positionwise Feed-
Forward Network to apply the two linear transformations with the ReLU activation in

between. Finally, the output of the FFN is added to the target and normalized as the
Decoder Layer’s output.

## Usage

### Training

1. Load the dataset and prepare the vocabulary:
    ```python
    src_sentences, tgt_sentences = load_data('OpenSubtitles.en-pl.en', 'OpenSubtitles.en-pl.pl.txt')

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    for sentence in src_sentences:
        src_vocab.add_sentence(sentence)
    for sentence in tgt_sentences:
        tgt_vocab.add_sentence(sentence)
    ```

2. Create the dataset and data loader:
    ```python
    train_dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
    ```

3. Initialize the model, optimizer, and loss function:
    ```python
    embed_size = 128
    num_heads = 8
    num_layers = 4
    ff_hidden_size = 256
    learning_rate = 0.0001
    dropout = 0.1

    model = Transformer(len(src_vocab), len(tgt_vocab), embed_size, num_heads, num_layers, ff_hidden_size, dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    ```

4. Train the model:
    ```python
    num_epochs = 1000
    for epoch in range(num_epochs):
        train_loss, epoch_losses_batch = train(model, train_loader, optimizer, criterion, device)
        epoch_losses.extend(epoch_losses_batch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}")
        torch.cuda.empty_cache()
        save_model(model, model_save_path)
    ```

### Translation

Use the trained model to translate sentences:

```python
def translate(model, sentence, src_vocab, tgt_vocab, device, max_len=100):
    model.eval()
    with torch.no_grad():
        tokens = tokenize(sentence)
        src_ids = torch.tensor(src_vocab.sentence_to_ids(["<sos>"] + tokens + ["<eos>"]), dtype=torch.long).unsqueeze(0).to(device)
        src_mask = create_src_mask(src_ids)

        tgt_ids = torch.tensor([tgt_vocab.token_to_id["<sos>"]], dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(max_len):
            tgt_mask = create_tgt_mask(tgt_ids)
            with torch.cuda.amp.autocast():
                output = model(src_ids, tgt_ids, src_mask, tgt_mask)
            next_token_id = output.argmax(dim=-1)[:, -1].item()
            if next_token_id == tgt_vocab.token_to_id["<eos>"]:
                break
            tgt_ids = torch.cat([tgt_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

        decoded_tokens = tgt_vocab.ids_to_sentence(tgt_ids.squeeze().cpu().tolist()[1:])
        words = decoded_tokens.split(' ')

        formatted_sentence = ''
        for i, word in enumerate(words):
            if i == 0:
                formatted_sentence += word.capitalize()
            elif word in {'.', ',', '!', '?'}:
                formatted_sentence += word
            else:
                formatted_sentence += ' ' + word

        return formatted_sentence
```

Example:
```python
translated_sentence = translate(model, "Hello, how are you?", src_vocab, tgt_vocab, device)
print(translated_sentence)
```

## License

This project is licensed under the MIT License.
