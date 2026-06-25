def deepgaze_density_loss(
    logits: torch.Tensor,
    fixation_map: torch.Tensor,
    eps: float = 1e-8,
    ) -> torch.Tensor:
    
    """
    Spatial cross-entropy between a target fixation density
    and the model's predicted fixation density.

    logits: B, 1, H, W], unnormalized model output.

    fixation_map: [B, 1, H, W], nonnegative smoothed fixation map.
    """

    if fixation_map.ndim == 3:  #Should be [B, 1 , 224, 224]
        fixation_map = fixation_map.unsqueeze(1)

    if logits.shape != fixation_map.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, "
            f"fixation_map={fixation_map.shape}"
        )

    fixation_map = fixation_map.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    """ Sum of target intensities for every image. -> Returns a value in the shape [B, 1, 1, 1] 
    i,e: [[[[sum1]]],[[[sum2]]],..] one val for each img in the batch """

    target_mass = fixation_map.sum(
        dim=(2, 3), # Across HxW - sum all values
        keepdim=True,
    )

    """ Convert fixation intensities into a probability distribution.
    Each image's target is now such that all pixels sums to 1. """
    target_density = fixation_map / target_mass.clamp_min(eps)


    """ Spatial log-softmax over H × W. 
    returns log(pi)
    which we then use for Loss as L = - sum_i (qi * log pi)
    qi is the target density while log(pi) is the log density
    """

    log_density = logits - torch.logsumexp(
        logits,
        dim=(2, 3),
        keepdim=True,
    )

    # Cross-entropy between target and predicted density.
    # sum over all values i where qi (target probability) is NOT zero
    loss_per_image = -(
        target_density * log_density
    ).sum(dim=(1, 2, 3))

    return loss_per_image.mean()
