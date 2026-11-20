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


def install_trl_static_padding(prompt_len, completion_len):
    """Trainium only: make TRL's own padding produce fixed shapes.

    TRL 0.29 tokenizes prompts without padding and then pads every batch itself with
    `trl.trainer.utils.pad` to the longest sequence in that batch, so the tokenizer patch above
    never applies and prompt length changes every step (= new graphs for generation, log-prob
    forward and backward). Here every left-padded tensor (prompts) is padded to `prompt_len` and
    every right-padded tensor (completions) to `completion_len`, via pad_to_multiple_of.
    """
    from trl.trainer import grpo_trainer as gt
    if getattr(gt.pad, "_bench_patched", False):
        return
    orig = gt.pad

    def pad(tensors, padding_value=0, padding_side="right", pad_to_multiple_of=None):
        if pad_to_multiple_of is None:
            longest = max((t.shape[0] for t in tensors), default=0)
            target = prompt_len if padding_side == "left" else completion_len
            if longest <= target:
                pad_to_multiple_of = target
        return orig(tensors, padding_value=padding_value, padding_side=padding_side,
                    pad_to_multiple_of=pad_to_multiple_of)

    pad._bench_patched = True
    gt.pad = pad
