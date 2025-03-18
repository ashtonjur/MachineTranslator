import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import re
from torch.nn.utils.rnn import pad_sequence
import os
import time
import datetime


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Selected device: {device}")

scaler = torch.cuda.amp.GradScaler()
accumulation_steps = 2

def tokenize(text):
    if isinstance(text, str):
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

def load_data(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as src_f, open(tgt_file, 'r', encoding='utf-8') as tgt_f:
        src_sentences = [["<sos>"] + tokenize(line.strip()) + ["<eos>"] for line in src_f.readlines()]
        tgt_sentences = [["<sos>"] + tokenize(line.strip()) + ["<eos>"] for line in tgt_f.readlines()]
    return src_sentences, tgt_sentences

def collate_fn(batch):
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_batch, tgt_batch

def create_tgt_mask(tgt):
    seq_len = tgt.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool()
    return mask.unsqueeze(0)

def create_src_mask(src, pad_token=0):
    return (src != pad_token).unsqueeze(1).unsqueeze(2)

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


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path, model):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")

model_save_path = "transformer_translation_model.pt"

def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_losses = []

    for i, (src, tgt) in enumerate(iterator):
        begin_time = time.time()
        ct = datetime.datetime.now()
        print(f"Start iteration {i} at {ct}")

        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_src_mask(src)
        tgt_mask = create_tgt_mask(tgt_input)

        with torch.cuda.amp.autocast():
            output = model(src, tgt_input, src_mask, tgt_mask)
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)

        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item()

        end_time = time.time()

        print(f"Iteration {i}, execution time: {end_time - begin_time}s\n")

    epoch_losses.append(epoch_loss / len(iterator))
    return epoch_loss / len(iterator), epoch_losses


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_sentences, tgt_sentences = load_data('OpenSubtitles.en-pl.en', 'OpenSubtitles.en-pl.pl.txt')

src_vocab = Vocabulary()
tgt_vocab = Vocabulary()

for sentence in src_sentences:
    src_vocab.add_sentence(sentence)
for sentence in tgt_sentences:
    tgt_vocab.add_sentence(sentence)

train_dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)

embed_size = 128
num_heads = 8
num_layers = 4
ff_hidden_size = 256
learning_rate = 0.0001
dropout = 0.1

model = Transformer(len(src_vocab), len(tgt_vocab), embed_size, num_heads, num_layers, ff_hidden_size, dropout).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)
epoch_losses = []

num_epochs = 1000
for epoch in range(num_epochs):
    train_loss, epoch_losses_batch = train(model, train_loader, optimizer, criterion, device)
    epoch_losses.extend(epoch_losses_batch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}")
    torch.cuda.empty_cache()
    save_model(model, model_save_path)

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

