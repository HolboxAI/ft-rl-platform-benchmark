"""Force fixed tensor shapes so Neuron compiles each graph once.

Neuron (XLA) compiles one graph per distinct input shape. Stock TRL pads each prompt batch to
its longest prompt, so every step has a different shape and Trainium recompiles. Here:

  * prompts are always padded to `max_prompt_length` (tokenizer patch below), and
  * generation always produces `max_completion_length` tokens (min_new_tokens in the config)
    with a static KV cache.

The same settings are applied on the GPU, so both platforms do exactly the same work.
"""


def force_prompt_padding(tokenizer, max_length):
    """Make padding=True / 'longest' pad to max_length for tensor outputs."""
    base_cls = tokenizer.__class__

    class _FixedPad(base_cls):
        def __call__(self, *args, **kwargs):
            if kwargs.get("return_tensors") is not None and kwargs.get("padding") in (True, "longest"):
                kwargs["padding"] = "max_length"
                kwargs["max_length"] = max_length
                kwargs["truncation"] = True
            return super().__call__(*args, **kwargs)

    _FixedPad.__name__ = base_cls.__name__
    tokenizer.__class__ = _FixedPad
    return tokenizer


def generation_kwargs(cfg, tokenizer):
    kw = {}
    if cfg.get("fixed_length_generation"):
        kw["min_new_tokens"] = cfg["max_completion_length"]
    if cfg.get("cache_implementation"):
        kw["cache_implementation"] = cfg["cache_implementation"]
    if tokenizer.pad_token_id is not None:
        kw["pad_token_id"] = tokenizer.pad_token_id
    return kw
