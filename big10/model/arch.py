from abc import ABC, abstractmethod

import torch

from big10.model.encoder.builder import build_matchup_tower
from big10.model.projector.builder import build_matchup_projector

from big10.constants import IGNORE_INDEX, MATCHUP_TOKEN_INDEX


class Big10MetaModel:

    def __init__(self, config):
        super(Big10MetaModel, self).__init__(config)

        if hasattr(config, "matchup_tower"):
            self.matchup_tower = build_matchup_tower(config, delay_loading=True)
            self.matchup_projector = build_matchup_projector(config)

    def get_matchup_tower(self):
        matchup_tower = getattr(self, "matchup_tower", None)
        if type(matchup_tower) is list:
            matchup_tower = matchup_tower[0]
        return matchup_tower

    def initialize_matchup_modules(self, model_args):
        matchup_tower = model_args.matchup_tower
        self.config.matchup_tower = matchup_tower

        if self.get_matchup_tower() is None:
            matchup_tower = build_matchup_tower(model_args)
            self.matchup_tower = matchup_tower
        else:
            matchup_tower = self.matchup_tower
            matchup_tower.load_model()

        self.config.use_matchup = True
        self.config.matchup_projector_type = getattr(
            model_args, "matchup_projector_type"
        )
        self.config.matchup_hidden_size = matchup_tower.hidden_size

        if getattr(self, "matchup_projector", None) is None:
            self.matchup_projector = build_matchup_projector(self.config)


class Big10MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_matchup_tower(self):
        return self.get_model().get_matchup_tower()

    def encode_matchups(self, matchups):
        matchup_features = self.get_model().get_matchup_tower()(matchups)
        # matchup_features is CausalLMOutputWithCrossAttentions
        # and we need to extract the last hidden states
        if hasattr(matchup_features, "hidden_states"):
            matchup_features = matchup_features.hidden_states[-1]
        matchup_features = self.get_model().matchup_projector(matchup_features)
        return matchup_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, matchups
    ):
        matchup_tower = self.get_matchup_tower()
        if matchup_tower is None or matchups is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and matchup_tower is not None
                and matchups is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(matchups) is list or matchups.ndim == 5:
            concat_matchups = torch.cat([matchup for matchup in matchups], dim=0)
            matchup_features = self.encode_matchups(concat_matchups)
            split_sizes = [matchup.shape[0] for matchup in matchups]
            matchup_features = torch.split(matchup_features, split_sizes, dim=0)
            matchup_features = [x.flatten(0, 1).to(self.device) for x in matchup_features]
        else:
            matchup_features = self.encode_matchups(matchups).to(self.device)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_matchup_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_matchups = (cur_input_ids == MATCHUP_TOKEN_INDEX).sum()
            if num_matchups == 0:
                cur_matchup_features = matchup_features[cur_matchup_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_matchup_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_matchup_idx += 1
                continue

            matchup_token_indices = (
                [-1]
                + torch.where(cur_input_ids == MATCHUP_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(matchup_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        matchup_token_indices[i] + 1 : matchup_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[matchup_token_indices[i] + 1 : matchup_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_matchups + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_matchups:
                    cur_matchup_features = matchup_features[cur_matchup_idx]
                    cur_matchup_idx += 1
                    cur_new_input_embeds.append(cur_matchup_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_matchup_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )
