"""Stratified dN/dS separation grid for the isolate CP-HMM runs (Ap + Pv).

Takes explicit ``LABEL=points.csv`` pairs (one per species) — needed because the
Ap and Pv runs both use the accession name ``snv_table`` and so cannot be told
apart by globbing. The layout is an aggregate row across all supplied species
followed by one row per species, with columns being the full-core / recombined /
clonal strata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dnds_dynamics.dnds.stratified import _recombination_theory_curve  # noqa: E402


DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "figures" / "isolate_cphmm_qpcore_stratified_dnds_grid"

STRATA = ["full_core", "recombined", "clonal"]
STRATUM_TITLES = {
    "full_core": "Full (core) genome",
    "recombined": "Recombined regions",
    "clonal": "Clonal regions",
}
STRATUM_COLORS = {
    "full_core": "tab:grey",
    "recombined": "#FF968D",
    "clonal": "#AECDE1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "points",
        nargs="*",
        default=[
            f"A. putredinis={PROJECT_ROOT / 'data' / 'isolate_dnds' / 'aputredinis' / 'stratified_dnds_points.csv'}",
            f"P. vulgatus={PROJECT_ROOT / 'data' / 'isolate_dnds' / 'pvulgatus' / 'stratified_dnds_points.csv'}",
        ],
        help="One or more LABEL=path/to/stratified_dnds_points.csv pairs, in row order "
             "(default: the vendored A. putredinis and P. vulgatus tables under data/isolate_dnds/).",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--aggregate-label", default="All isolates")
    parser.add_argument("--fig-width", type=float, default=6.9,
                        help="Figure width in inches (before tight bbox).")
    parser.add_argument("--row-height", type=float, default=1.15,
                        help="Per-row height in inches.")
    parser.add_argument("--png-dpi", type=int, default=300)
    return parser.parse_args()


def parse_points_args(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=path, got: {spec!r}")
        label, path = spec.split("=", 1)
        out.append((label.strip(), Path(path).expanduser()))
    return out


def load_points(entries: list[tuple[str, Path]]) -> pd.DataFrame:
    dfs = []
    for label, path in entries:
        if not path.exists():
            raise FileNotFoundError(f"Stratified points file not found for {label!r}: {path}")
        df = pd.read_csv(path)
        df["row_label"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)


def plot_grid(
    points: pd.DataFrame,
    labels: list[str],
    output_prefix: Path,
    aggregate_label: str,
    png_dpi: int,
    fig_width: float = 6.9,
    row_height: float = 1.15,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(aggregate_label, points)] + [
        (label, points[points["row_label"] == label]) for label in labels
    ]
    n_rows = len(rows)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7}):
        fig, axes = plt.subplots(
            n_rows,
            len(STRATA),
            figsize=(fig_width, max(2.2, row_height * n_rows)),
            dpi=png_dpi,
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        fig.subplots_adjust(left=0.25, right=0.99, top=0.94, bottom=0.12, hspace=0.22, wspace=0.08)

        d_s_arr, dnds_arr = _recombination_theory_curve()
        for row_idx, (row_label, row_points) in enumerate(rows):
            for col_idx, stratum in enumerate(STRATA):
                ax = axes[row_idx, col_idx]
                sub = row_points[(row_points["stratum"] == stratum) & row_points["plot_valid"]]
                is_aggregate = row_idx == 0
                ax.scatter(
                    sub["plot_dS"],
                    sub["dNdS"],
                    s=4 if is_aggregate else 2,
                    alpha=0.30 if is_aggregate else 0.42,
                    color=STRATUM_COLORS[stratum],
                    linewidths=0,
                    rasterized=True,
                )
                if stratum == "full_core":
                    ax.plot(d_s_arr, dnds_arr, linestyle="-", color="k", linewidth=0.45)
                ax.axhline(1, linewidth=0.45, linestyle="--", color="0.55")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(2e-6, 2e-2)
                ax.set_ylim(1e-2, 1e1)
                if row_idx == 0:
                    ax.set_title(STRATUM_TITLES[stratum])
                if col_idx > 0:
                    ax.tick_params(labelleft=False)
                if row_idx < n_rows - 1:
                    ax.tick_params(labelbottom=False)
                if col_idx == 0:
                    ax.text(
                        -0.38,
                        0.5,
                        row_label,
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=6.5,
                    )
                    ax.set_ylabel("$dN/dS$")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("$dS$ (Full genome)")

        png_path = output_prefix.with_suffix(".png")
        pdf_path = output_prefix.with_suffix(".pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    metadata = {
        "labels": labels,
        "num_points": int(points.shape[0]),
        "num_valid_points": int(points["plot_valid"].sum()),
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    entries = parse_points_args(args.points)
    labels = [label for label, _ in entries]
    points = load_points(entries)
    plot_grid(points, labels, args.output_prefix, args.aggregate_label, args.png_dpi,
              fig_width=args.fig_width, row_height=args.row_height)
    print(f"wrote {args.output_prefix.with_suffix('.png')}")
    print(f"wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
