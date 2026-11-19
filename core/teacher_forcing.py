"""Input construction for teacher-forced training - a port of ``build_teacher_forcing_input`` from Alexandria's
``train_lora.py`` (MIT, github.com/Finrandojin/alexandria-audiobook; the code itself mirrors how Qwen3-TTS builds its
input inside ``generate()``).

For one training sample the sequence is, in order:

* role: the first 3 text tokens (``<|im_start|>assistant\\n``);
* codec prefix ``[think, think_bos, language_id, think_eos]`` (without a language: ``[nothink, think_bos, think_eos]``);
* speaker embedding, ``codec_pad``, ``codec_bos``;
* text ``text_ids[:, 3:-5]`` + ``tts_eos``, with ``codec_pad`` embeddings added on top;
* end of prefill: ``tts_pad + codec_bos``;
* audio steps: the sum of the embeddings of all 16 codec groups + ``tts_pad``;
* one final step carrying the ``codec_eos`` label (end of speech).

Labels: the first codec group on the audio steps, ``codec_eos`` on the final step, ``-100`` (ignored) on the prefill.  Apart from the end-of-speech label the input is the same as in
Alexandria; adapters trained here still load unchanged in Alexandria.
"""
from __future__ import annotations

from typing import Any, Tuple


def build_assistant_text(text: str) -> str:
    """Wrap ``text`` in the chat template the Qwen3-TTS tokenizer expects (same as ``train_lora.py``)."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_teacher_forcing_input(sample: dict, hf_model: Any, talker: Any, device: Any,
                                language: str = "russian") -> Tuple[Any, Any, Any, int]:
    """Build ``(inputs_embeds [1, prefill+T, D], labels [1, prefill+T], all_codec_ids [T, G], prefill_len)``.

    ``sample`` holds ``codec_ids`` ([T, G]), ``spk_embedding`` ([1, enc_dim]) and ``text_ids`` ([1, L]).  ``talker`` is the
    original (not peft-wrapped) talker module; ``hf_model.config`` supplies the ``tts_*_token_id`` values.  ``language``
    selects the codec language id (an unknown language falls back to the "nothink" prefix).
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

    # One extra step whose only job is to carry the end-of-speech label: the model must learn to emit ``codec_eos``
    # after the last frame.  Without it (as in the original script) the fine-tuned talker never learns to stop, and
    # from ~3e-6 x 20 epochs on it babbles until the token limit (found in the first real-voice test).
    eos_embed = tts_pad_embed.squeeze(0) + emb(torch.tensor([[tc.codec_eos_token_id]], device=device,
                                                            dtype=text_ids.dtype)).squeeze(0)
    full_input = torch.cat([prefill_embeds, audio_embeds, eos_embed.unsqueeze(0).to(audio_embeds.dtype)], dim=1)
    labels = torch.full((1, prefill_len + T + 1), -100, device=device, dtype=torch.long)
    labels[0, prefill_len:prefill_len + T] = codec_ids_2d[:, 0]
    labels[0, prefill_len + T] = tc.codec_eos_token_id
    return full_input, labels, codec_ids_2d, prefill_len
