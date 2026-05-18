import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalizes non-zero values in an image array using z-score normalization.

    This function performs z-score normalization (subtract mean, divide by standard deviation) for every image individually.
    Mean and standard deviation are calculated only from non-zero values.
    Zero values are preserved in the output.

    Args:
        x: A numpy array to be normalized. The normalization is applied along the
           last two dimensions (-1, -2).

    Returns:
        A numpy array of the same shape as the input, where non-zero values have been
        normalized. Zero values remain unchanged.

    Note:
        This function treats zero values specially. They are excluded from the mean and
        standard deviation calculations and are not modified in the output.
    """
    x_norm = np.copy(x).astype(float)
    non_zero_mask = (x != 0)

    x_nanzero = np.copy(x).astype(float)
    x_nanzero[x_nanzero == 0] = np.nan

    mean = np.nanmean(x_nanzero, axis=(-1, -2), keepdims=True)
    std = np.nanstd(x_nanzero, axis=(-1, -2), keepdims=True)

    normalized = (x - mean) / std
    x_norm = np.where(non_zero_mask, normalized, x_norm)

    return x_norm


def denormalize(x: np.ndarray, dim_x: int = -1, dim_y: int = -2) -> np.ndarray:
    """Denormalizes an image array that has been normalized using a specific scaling method.

    This function reverses the normalization process by scaling the non-zero values of the
    input array based on the minimum and maximum non-zero values in each image.

    Non-zero values are rescaled to a range of [0, 255]. Zero values are preserved in the output.

    Args:
        x: A numpy array to be denormalized.
        dim_x: The dimension along which to denormalize (default: -1).
        dim_y: The dimension along which to denormalize (default: -2).
    Returns:
        A numpy array of the same shape as the input, where non-zero values have been
        denormalized. Zero values remain unchanged.

    Note:
        This function treats zero values specially. They are excluded from the scaling
        calculations and are not modified in the output.
    """
    x_unnorm = np.copy(x).astype(float)
    non_zero_mask = (x != 0)

    x_nanzero = np.copy(x).astype(float)
    x_nanzero[x_nanzero == 0] = np.nan

    x_nanzero = x_nanzero - np.nanmin(x_nanzero, axis=(dim_x, dim_y), keepdims=True) + 1e-8
    x_nanzero = x_nanzero * 255 / np.nanmax(x_nanzero, axis=(dim_x, dim_y), keepdims=True)

    x_unnorm = np.where(non_zero_mask, x_nanzero, x_unnorm)

    return x_unnorm
