# Transformer-based Machine Translation

This repository contains a PyTorch implementation of a Transformer model for machine translation. The model is trained to translate sentences from English to Polish using the OpenSubtitles dataset.

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
    num_epochs = 100
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
