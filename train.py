import torch
from big10.preprocessing.load import load_and_preprocess_data
from tqdm import tqdm
from big10.preprocessing.context import get_game_context, extract_game_context
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
from big10.model import Big10LlamaConfig
from big10.model.builder import load_pretrained_model
from big10.preprocessing.dataset import Big10Dataset
from transformers import Trainer, TrainingArguments, GPT2Config


class Big10Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        matchups = inputs.pop("matchups")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Create labels for autoregressive language modeling
        labels = input_ids.clone()
        labels = torch.roll(labels, shifts=-1, dims=1)
        labels[:, -1] = -100  # Ignore loss for the last token

        # Set -100 for padding tokens to ignore them in loss computation
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, matchups=matchups, labels=labels)
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
        

def preprocess_dataset(game_ids):
    training_data = []
    for id in tqdm(game_ids):
        try:
            game_context = get_game_context(id)
            if game_context is not None:
                game_data = extract_game_context(game_context)
                training_data.append(game_data)
        except Exception as e:
            continue
    return training_data

def train_encoder_tokenizer(data, vocab_size=512, max_length=512):
    text = ""
    for game in data:
        vals = [str(v) for k, v in game.items() if k != "drives"]
        text += " ".join(vals)
    special_tokens = ["<bos>", "<pad>", "<eos>", "<unk>"]
    sp_tokenizer = SentencePieceBPETokenizer()
    sp_tokenizer.train_from_iterator(
        text,
        vocab_size=vocab_size,
        min_frequency=1,
        show_progress=True,
        special_tokens=special_tokens
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=sp_tokenizer._tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        max_length=max_length
    )
    return tokenizer

def main():
    _, games, _, _, _ = load_and_preprocess_data()
    game_ids = games.select("id").to_pandas()["id"].tolist()
    training_data = preprocess_dataset(game_ids)
    
    base_model_path = "meta-llama/Llama-3.2-1B"

    # Define Big10 config
    big10_config = Big10LlamaConfig.from_pretrained(base_model_path)
    big10_config.max_length = 4096
    big10_config.matchup_tower = "gpt2"
    big10_config.encoder_config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_ctx=128,
        n_embd=256,
        n_layer=6,
        n_head=8,
        initializer_range=0.02,
        device="mps",
        output_hidden_states=True
    )

    # Train encoder tokenizer
    encoder_tokenizer = train_encoder_tokenizer(training_data, big10_config.encoder_config.vocab_size, big10_config.encoder_config.n_positions)

    # Initialize Big10 model
    tokenizer, model = load_pretrained_model(
        model_path=base_model_path,
        model_type="llama-3.2",
        device="mps",
        config=big10_config,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<drive>", "</drive>"]})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    # Create Big10 dataset
    dataset = Big10Dataset(
        data=training_data,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=tokenizer,
        encoder_max_length=big10_config.encoder_config.n_positions,
        decoder_max_length=big10_config.max_length
    )

    training_args = TrainingArguments(
        output_dir="big10-llama",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=1000,
        save_total_limit=1,
        optim="adamw_8bit",
    )

    trainer = Big10Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
        

if __name__ == "__main__":
    main()