"""Search for a 24-microphone geometry with diverse pairwise distances.

The historical array consists of three orthogonal branches with eight
microphones per branch. Candidate geometries are generated randomly under
the original spacing constraints and scored by the number of distinct
pairwise distances.

The search is intended as a geometry-design tool. It does not modify the
fixed array24 geometry used by the acoustic-analysis pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np

from acoustic_estimation.geometry import (
    optimize_array24_geometry,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Randomly search for a 24-microphone geometry "
            "maximising the number of distinct pairwise distances."
        )
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help=(
            "Maximum number of random geometries to test "
            "(default: 10000)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for reproducible searches."
        ),
    )

    parser.add_argument(
        "--length",
        type=int,
        default=120,
        help=(
            "Length of each branch in centimetres "
            "(default: 120)."
        ),
    )

    parser.add_argument(
        "--min-spacing",
        type=int,
        default=2,
        help=(
            "Minimum adjacent-microphone spacing in centimetres "
            "(default: 2)."
        ),
    )

    parser.add_argument(
        "--max-spacing",
        type=int,
        default=15,
        help=(
            "Maximum adjacent-microphone spacing in centimetres "
            "(default: 15)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the historical array24 geometry search."""
    args = parse_args()

    result = optimize_array24_geometry(
        n_iterations=args.iterations,
        seed=args.seed,
        length_x_cm=args.length,
        length_y_cm=args.length,
        length_z_cm=args.length,
        min_spacing_cm=args.min_spacing,
        max_spacing_cm=args.max_spacing,
    )

    theoretical_maximum = (
        24 * 23 // 2
    )

    percentage = (
        100.0
        * result.distinct_distance_count
        / theoretical_maximum
    )

    print("=" * 72)
    print("24-MICROPHONE GEOMETRY SEARCH")
    print("=" * 72)

    print(
        f"Iterations completed     : "
        f"{result.iterations_completed}"
    )

    print(
        f"Distinct distances       : "
        f"{result.distinct_distance_count}"
    )

    print(
        f"Theoretical maximum      : "
        f"{theoretical_maximum}"
    )

    print(
        f"Fraction of maximum      : "
        f"{percentage:.2f}%"
    )

    print()
    print("Best geometry")
    print("-------------")

    print(
        f"X (cm) : "
        f"{result.x_cm}"
    )

    print(
        f"gaps   : "
        f"{np.diff(result.x_cm)}"
    )

    print()

    print(
        f"Y (cm) : "
        f"{result.y_cm}"
    )

    print(
        f"gaps   : "
        f"{np.diff(result.y_cm)}"
    )

    print()

    print(
        f"Z (cm) : "
        f"{result.z_cm}"
    )

    print(
        f"gaps   : "
        f"{np.diff(result.z_cm)}"
    )

    print()
    print("Boundary distances")
    print("------------------")

    print(
        f"X : left={result.x_cm[0]} cm, "
        f"right={args.length - result.x_cm[-1]} cm"
    )

    print(
        f"Y : left={result.y_cm[0]} cm, "
        f"right={args.length - result.y_cm[-1]} cm"
    )

    print(
        f"Z : left={result.z_cm[0]} cm, "
        f"right={args.length - result.z_cm[-1]} cm"
    )


if __name__ == "__main__":
    main()