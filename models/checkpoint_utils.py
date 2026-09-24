import torch


def load_checkpoint_compat(fpath, map_location="cpu"):
    return torch.load(fpath, map_location=map_location, weights_only=False)


def load_state_dict_checked(
    module,
    state_dict,
    *,
    allowed_missing=(),
    allowed_unexpected=(),
    context="checkpoint",
):
    """Load flexibly but fail when incompatibilities exceed known regenerated buffers."""
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    invalid_missing = sorted(set(missing) - set(allowed_missing))
    invalid_unexpected = sorted(set(unexpected) - set(allowed_unexpected))
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(
            f"Incompatible {context}: missing={invalid_missing}, "
            f"unexpected={invalid_unexpected}"
        )
    if missing or unexpected:
        print(
            f"Accepted known {context} differences: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return missing, unexpected
