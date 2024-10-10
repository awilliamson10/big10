from torch.utils.data import Dataset

class Big10Dataset(Dataset):
    def __init__(self, data, encoder_tokenizer, decoder_tokenizer, encoder_max_length=512, decoder_max_length=128):
        self.data = data
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoder_vals = [str(v) for k, v in item.items() if k != 'drives']
        matchup = self.encoder_tokenizer(" ".join(encoder_vals), max_length=self.encoder_max_length, padding="max_length", truncation=True, return_tensors="pt")
        drives = self.decoder_tokenizer(item['drives'], max_length=self.decoder_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": drives.input_ids.squeeze(),
            "attention_mask": drives.attention_mask.squeeze(),
            "matchups": matchup.input_ids.squeeze()
        }