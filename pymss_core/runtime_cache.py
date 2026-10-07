"""Lifecycle cleanup for derived model tensors, independent of inference backend."""


def _clear_cache_dicts(module, *names):
    for name in names:
        cache = getattr(module, name, None)
        if isinstance(cache, dict): cache.clear()


def clear_model_runtime_caches(model):
    """Release known inference caches and module-owned hooks without changing weights."""
    cache_names = (
        "_pymss_cos_sin_cache", "_pymss_apollo_inference_cache",
        "_pymss_mlx_full_param_cache", "_pymss_mlx_cos_sin_cache", "_pymss_mlx_attention_cache",
        "_pymss_mlx_feed_forward_cache", "_pymss_mlx_norm_cache", "_pymss_mlx_full_band_split_cache",
        "_pymss_mlx_full_mask_cache", "_pymss_mlx_full_mbr_cache",
        "_pymss_mlx_compiled_attention_cache", "_pymss_mlx_compiled_feed_forward_cache",
    )
    first_error = None
    for module in model.modules():
        clear = getattr(module, "clear_runtime_cache", None)
        if callable(clear):
            try: clear()
            except Exception as error:
                if first_error is None: first_error = error
        _clear_cache_dicts(module, *cache_names)
        if hasattr(module, "_pymss_group_cache_warm_key"): module._pymss_group_cache_warm_key = None
    if first_error is not None: raise first_error
