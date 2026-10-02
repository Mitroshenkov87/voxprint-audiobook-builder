"""Вход для обучения с teacher forcing - порт build_teacher_forcing_input из Alexandria train_lora.py
(MIT, github.com/Finrandojin/alexandria-audiobook; сам код повторяет сборку входа в generate() Qwen3-TTS).

Для одного примера:
  * role: первые 3 токена текста (<|im_start|>assistant\\n);
  * codec-префикс [think, think_bos, language_id, think_eos] (без языка: [nothink, think_bos, think_eos]);
  * эмбеддинг диктора, codec_pad, codec_bos;
  * текст text_ids[:, 3:-5] + tts_eos, поверх - эмбеддинги codec_pad;
  * конец префилла: tts_pad + codec_bos;
  * аудио-шаги: сумма эмбеддингов всех 16 кодовых групп + tts_pad.
Метки: первая кодовая группа на аудио-шагах, -100 на префилле.
"""
from __future__ import annotations

from typing import Any, Tuple


def build_assistant_text(text: str) -> str:
    """Шаблон текста, как в train_lora.py."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_teacher_forcing_input(sample: dict, hf_model: Any, talker: Any, device: Any,
                                language: str = "russian") -> Tuple[Any, Any, Any, int]:
    """Возвращает (inputs_embeds [1, prefill+T, D], labels [1, prefill+T], all_codec_ids [T, G], prefill_len).

    `talker` - исходный (не обёрнутый peft) talker; `hf_model.config` берётся для tts_*_token_id.
    """
    import torch

    config = hf_model.config
    tc = config.talker_config
    codec_ids_2d = sample["codec_ids"]          # [T, G]
    spk_embedding = sample["spk_embedding"]     # [1, enc_dim]
    text_ids = sample["text_ids"]               # [1, L]
    T = codec_ids_2d.shape[0]
    num_code_groups = tc.num_code_groups

    special_ids = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
                               device=device, dtype=text_ids.dtype)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(special_ids)).chunk(3, dim=1)

    role_embed = talker.text_projection(talker.get_text_embeddings()(text_ids[:, :3]))

    language_id = tc.codec_language_id.get(language, None) if tc.codec_language_id else None
    if language_id is not None:
        codec_prefill = [[tc.codec_think_id, tc.codec_think_bos_id, language_id, tc.codec_think_eos_id]]
    else:
        codec_prefill = [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]]
    emb = talker.get_input_embeddings()
    codec_prefix_embed = emb(torch.tensor(codec_prefill, device=device, dtype=text_ids.dtype))
    codec_suffix_embed = emb(torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]], device=device, dtype=text_ids.dtype))
    codec_embed = torch.cat([codec_prefix_embed, spk_embedding.view(1, 1, -1).to(codec_prefix_embed.dtype),
                             codec_suffix_embed], dim=1)
    prefix_codec_len = codec_embed.shape[1]
    tts_prefix = torch.cat([tts_pad_embed.expand(-1, prefix_codec_len - 2, -1), tts_bos_embed], dim=1)
    prefix_embed = tts_prefix + codec_embed[:, :-1]
    parts = [torch.cat([role_embed, prefix_embed], dim=1)]

    text_content_ids = text_ids[:, 3:-5]
    text_content_len = text_content_ids.shape[1]
    text_content_embed = talker.text_projection(talker.get_text_embeddings()(text_content_ids))
    text_with_eos = torch.cat([text_content_embed, tts_eos_embed], dim=1)
    text_pad_ids = torch.full((1, text_content_len + 1), tc.codec_pad_id, device=device, dtype=text_ids.dtype)
    parts.append(text_with_eos + emb(text_pad_ids))

    codec_bos_embed = emb(torch.tensor([[tc.codec_bos_id]], device=device, dtype=text_ids.dtype))
    parts.append(tts_pad_embed + codec_bos_embed)
    prefill_embeds = torch.cat(parts, dim=1)
    prefill_len = prefill_embeds.shape[1]

    group_embeds = [emb(codec_ids_2d[:, :1])]
    for g in range(1, num_code_groups):
        group_embeds.append(talker.code_predictor.get_input_embeddings()[g - 1](codec_ids_2d[:, g:g + 1]))
    codec_sum = torch.cat(group_embeds, dim=1).sum(dim=1)                   # [T, D]
    audio_embeds = (codec_sum + tts_pad_embed.squeeze(0)).unsqueeze(0)      # [1, T, D]

    full_input = torch.cat([prefill_embeds, audio_embeds], dim=1)
    labels = torch.full((1, prefill_len + T), -100, device=device, dtype=torch.long)
    labels[0, prefill_len:] = codec_ids_2d[:, 0]
    return full_input, labels, codec_ids_2d, prefill_len
