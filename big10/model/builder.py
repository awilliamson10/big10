import torch

from transformers import AutoTokenizer, BitsAndBytesConfig, logging

logging.set_verbosity_error()

from big10.model import Big10LlamaForCausalLM, Big10LlamaConfig


def load_pretrained_model(model_path, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", **kwargs):
    if model_type not in {'llama-3.2'}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = Big10LlamaForCausalLM.from_pretrained(model_path, **kwargs)

    model.resize_token_embeddings(len(tokenizer))

    matchup_tower = model.get_matchup_tower()
    if not matchup_tower.is_loaded:
        matchup_tower.load_model()
    matchup_tower.to(device=device, dtype=torch.float16)

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model