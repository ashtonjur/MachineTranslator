class Vocabulary:
    def __init__(self):
        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        self.id_to_token = {0: "<pad>", 1: "<unk>"}
        self.idx = 2

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.token_to_id:
                self.token_to_id[word] = self.idx
                self.id_to_token[self.idx] = word
                self.idx += 1

    def sentence_to_ids(self, sentence):
        return [self.token_to_id.get(word, 1) for word in sentence]

    def ids_to_sentence(self, ids):
        return ' '.join([self.id_to_token.get(id, "<unk>") for id in ids])

    def __len__(self):
        return len(self.token_to_id)
