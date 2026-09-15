#!/usr/bin/env python3

import argparse
import os
import joblib
import numpy as np

from gs_dist import compute_gs_dist


ALPHA = 5.0

FEATURE_NAMES = [
    "position",
    "sh_dc_yuv_411",
    "sh_ac_y",
    "sh_ac_uv",
]


def compute_gspqm(reference_path, distorted_path, model_path, verbose=False):
    """
    Compute GS-PQM between an uncompressed reference GS model and a
    compressed/distorted GS model.

    Parameters
    ----------
    reference_path : str
        Path to the reference 3DGS PLY file.

    distorted_path : str
        Path to the compressed/distorted 3DGS PLY file.

    model_path : str
        Path to the pretrained GS-PQM model.

    Returns
    -------
    predicted_dmos : float
        GS-PQM predicted DMOS.

    features : dict
        GS-Dist errors used by the regression model.
    """

    # Compute GS-Dist errors.
    gs_dist = compute_gs_dist(
        reference_path,
        distorted_path,
        alpha=ALPHA,
        verbose=verbose,
    )

    # Construct the feature vector in exactly the same order
    # used to train GS-PQM.
    features = np.array([[
        gs_dist["position"],
        gs_dist["sh_dc_yuv_411"],
        gs_dist["sh_ac_y"],
        gs_dist["sh_ac_uv"],
    ]], dtype=np.float64)

    # Load the pretrained scaler + SVR pipeline.
    model = joblib.load(model_path)

    # Predict perceptual quality.
    predicted_dmos = float(model.predict(features)[0])

    feature_dict = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, features[0])
    }

    return predicted_dmos, feature_dict


def main():
    parser = argparse.ArgumentParser(description=("GS-PQM: Full-reference perceptual quality metric for compressed Gaussian Splatting models."))
    parser.add_argument("--ref", required=True, help="Path to the uncompressed reference GS PLY file.")
    parser.add_argument("--dist", required=True, help="Path to the compressed/distorted GS PLY file.")
    parser.add_argument("--verbose", action="store_true", help="Display GS-Dist errors and neighborhood statistics.")
    default_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gspqm_model.pkl")
    parser.add_argument("--model", default=default_model, help="Path to the pretrained GS-PQM model.")
    args = parser.parse_args()

    predicted_dmos, features = compute_gspqm(args.ref, args.dist, args.model, verbose=args.verbose)

    if args.verbose:
        print("\nGS-Dist errors:")
        for name, value in features.items():
            print(f"  {name:<20} {value:.12e}")

    print(f"\nGS-PQM predicted DMOS: {predicted_dmos:.6f}")


if __name__ == "__main__":
    main()