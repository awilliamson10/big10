import torch
from transformers import Trainer, TrainingArguments
from big10.model import Big10LlamaConfig
from big10.model.builder import load_pretrained_model
from big10.preprocessing.dataset import Big10Dataset
from big10.preprocessing.context import load_and_preprocess_data
from typing import Dict

def format_examples(examples, tokenizer):
    data = []
    for example in examples:
        tokenized_drives = tokenizer(example["drives_text"], padding="max_length", truncation=True, return_tensors="pt")
        data.append({
            "matchups": example["game_input"],
            "input_ids": tokenized_drives["input_ids"].squeeze(0),
            "attention_mask": tokenized_drives["attention_mask"].squeeze(0),
            "labels": tokenized_drives["input_ids"].squeeze(0)
        })

def main():
    # Load and preprocess data
    stats, games, drives, weather, winprobs = load_and_preprocess_data()

    # Get all game IDs
    game_ids = games["id"].to_list()
    print(f"Loaded {len(game_ids)} game IDs")

    # Create dataset
    dataset = Big10Dataset(game_ids)
    # preprocess dataset


    # Load pre-trained LLaMA model
    base_model_path = "meta-llama/Llama-3.2-1B"
    
    # Define Big10 config
    big10_config = Big10LlamaConfig.from_pretrained(base_model_path)
    big10_config.matchup_tower = "encoder"
    big10_config.num_venues = games["venueId"].n_unique()
    big10_config.num_teams = stats["team"].n_unique()
    big10_config.num_weather_conditions = weather["weatherCondition"].n_unique()
    big10_config.num_conferences = stats["conference"].n_unique()
    big10_config.num_numerical_features = len(dataset[0]["game_input"]) - 16

    # Initialize Big10 model
    tokenizer, model = load_pretrained_model(base_model_path, "llama-3.2", config=big10_config, device="mps")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<drive>", "</drive>"]})
    tokenizer.add_tokens
    model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./big10_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        use_mps_device=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=big10_data_collator(tokenizer)
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()