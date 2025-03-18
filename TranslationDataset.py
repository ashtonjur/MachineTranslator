class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, pad_token=0):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad_token = pad_token
    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        src_ids = torch.tensor(self.src_vocab.sentence_to_ids(src_sentence), dtype=torch.long)
        tgt_ids = torch.tensor(self.tgt_vocab.sentence_to_ids(tgt_sentence), dtype=torch.long)

        return src_ids, tgt_ids
