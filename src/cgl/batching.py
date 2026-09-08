"""Real batched generation with an independent seeded token sampler for each sequence."""

import torch
from transformers import LogitsProcessor

from cgl.artifacts import digest


class IndependentTokenSampler(LogitsProcessor):
    def __init__(self, seeds, *, device, temperature=1.0, top_p=1.0):
        self.generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        self.temperature, self.top_p = temperature, top_p

    def __call__(self, input_ids, scores):
        output = torch.full_like(scores, -torch.inf)
        for index, generator in enumerate(self.generators):
            logits = scores[index].float() / self.temperature
            if self.top_p < 1:
                sorted_logits, order = logits.sort(descending=True)
                exclude = sorted_logits.softmax(-1).cumsum(-1) > self.top_p
                exclude[1:] = exclude[:-1].clone()
                exclude[0] = False
                logits = logits.scatter(0, order, sorted_logits.masked_fill(exclude, -torch.inf))
            token = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            output[index, token] = 0
        return output


def generate_batch(
    model, tokenizer, messages, seeds, *, max_new_tokens=256, temperature=1.0, top_p=1.0, prefix=""
):
    rendered = [
        tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True) + prefix
        for turns in messages
    ]
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            rendered, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(model.device)
    finally:
        tokenizer.padding_side = previous_padding
    processors = (
        [IndependentTokenSampler(seeds, device=model.device, temperature=temperature, top_p=top_p)]
        if temperature > 0
        else []
    )
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            logits_processor=processors,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results = []
    width = encoded["input_ids"].shape[1]
    for index, sequence in enumerate(output):
        tokens = sequence[width:].tolist()
        finished = next((i for i, token in enumerate(tokens) if token in eos), None)
        if finished is not None:
            tokens = tokens[: finished + 1]
        prompt_tokens = encoded["input_ids"][index][
            encoded["attention_mask"][index].bool()
        ].tolist()
        results.append(
            {
                "response": prefix + tokenizer.decode(tokens, skip_special_tokens=True),
                "generated_token_ids": tokens,
                "prompt_token_ids": prompt_tokens,
                "prompt_render_sha256": digest(rendered[index]),
                "truncated": finished is None and len(tokens) >= max_new_tokens,
                "finished_with_eos": finished is not None,
            }
        )
    return results
