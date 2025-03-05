import torch


def _gaudi_init_added_embeddings_weights_with_mean(
    self, old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
):
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/modeling_utils.py#L2406
    Changes:
    - torch.linalg.eigvals is not supported on HPU so run it on CPU
    """
    old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
    mean_embeddings = torch.mean(old_embeddings_weight, axis=0)
    old_centered_embeddings = old_embeddings_weight - mean_embeddings
    covariance = old_centered_embeddings.T @ old_centered_embeddings / old_num_tokens

    # Check if the covariance is positive definite.
    # TODO: do not move `covariance` to the host once torch.linalg.eigvals is supported on HPU
    eigenvalues = torch.linalg.eigvals(covariance.to("cpu"))
    is_covariance_psd = bool(
        (covariance == covariance.T).all() and not torch.is_complex(eigenvalues) and (eigenvalues > 0).all()
    )
    if is_covariance_psd:
        # If covariances is positive definite, a distribution can be created. and we can sample new weights from it.
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            mean_embeddings, covariance_matrix=1e-9 * covariance
        )
        new_embeddings.weight.data[-1 * added_num_tokens :, :] = distribution.sample(
            sample_shape=(added_num_tokens,)
        ).to(old_embeddings.weight.dtype)
    else:
        # Otherwise, just initialize with the mean. because distribtion will not be created.
        new_embeddings.weight.data[-1 * added_num_tokens :, :] = (
            mean_embeddings[None, :].repeat(added_num_tokens, 1).to(old_embeddings.weight.dtype)
        )
