"""Render v2 COVERT sensitivity figures from completed aggregate CSVs only."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensitivity_analysis._config import EXPERIMENT_DEFINITIONS


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "figures"
F1_COLOR = "#55A9DC"  # RGB (85, 169, 220)
PRECISION_COLOR = "#D9656D"  # RGB (217, 101, 109)
RECALL_COLOR = "#4FA66F"  # RGB (79, 166, 111)
DEFAULT_COLOR = "#B7323A"
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "reference_blue", ("#E3F4FD", F1_COLOR, "#236FA8")
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate COVERT sensitivity figures from formal Main8 summaries."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--scope",
        choices=("heldout_excluding_dev10", "all_main8"),
        default="heldout_excluding_dev10",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Diagnostic use only; visibly marks panels with incomplete sweeps.",
    )
    return parser


def _metric_columns(scope: str) -> tuple[str, str, str]:
    prefix = "heldout" if scope == "heldout_excluding_dev10" else "all_main8"
    return (
        f"{prefix}_precision_cb",
        f"{prefix}_recall_cb",
        f"{prefix}_f1_cb",
    )


def _read_summary(
    experiment_name: str, *, scope: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    definition = EXPERIMENT_DEFINITIONS[experiment_name]
    path = ROOT / definition["directory"] / "results" / "summary.csv"
    expected_ids = [str(item["id"]) for item in definition["configurations"]]
    if not path.is_file():
        return [], {"complete": False, "observed": 0, "expected": len(expected_ids), "missing": expected_ids}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    precision_field, recall_field, f1_field = _metric_columns(scope)
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        config_id = str(raw.get("config_id") or "")
        if config_id in by_id:
            raise RuntimeError(f"Duplicate config_id in {path}: {config_id}")
        if config_id not in expected_ids:
            raise RuntimeError(f"Unexpected config_id in {path}: {config_id}")
        if any(
            raw.get(field) in (None, "")
            for field in (precision_field, recall_field, f1_field)
        ):
            continue
        precision = float(raw[precision_field])
        recall = float(raw[recall_field])
        f1 = float(raw[f1_field])
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (precision, recall, f1)
        ):
            raise ValueError(f"Invalid Precision/Recall/F1 for {experiment_name}/{config_id}")
        values: dict[str, Any] = {
            "config_id": config_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        for column in definition["parameter_columns"]:
            text = str(raw.get(column) or "")
            if column == "surface_variation_scales":
                import json

                values[column] = tuple(int(value) for value in json.loads(text))
            elif column in {"graph_knn_k", "theta_ac_deg", "h_max"}:
                values[column] = int(float(text))
            else:
                values[column] = float(text)
        by_id[config_id] = values
    rows = [by_id[config_id] for config_id in expected_ids if config_id in by_id]
    missing = [config_id for config_id in expected_ids if config_id not in by_id]
    return rows, {
        "complete": not missing,
        "observed": len(rows),
        "expected": len(expected_ids),
        "missing": missing,
    }


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _score_limits(values: Sequence[float], *, include_zero: bool = False) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    low, high = min(finite), max(finite)
    if include_zero or low < 0.10:
        return 0.0, min(1.0, high + max(0.04, 0.08 * max(high, 0.1)))
    span = max(high - low, 0.03)
    padding = max(0.012, 0.12 * span)
    return max(0.0, low - padding), min(1.0, high + padding)


def _format_score_axis(axis: plt.Axes, values: Sequence[float], *, include_zero: bool = False) -> None:
    axis.set_ylim(*_score_limits(values, include_zero=include_zero))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(color="0.88", linewidth=0.6)


def _default_id(experiment_name: str) -> str:
    definition = EXPERIMENT_DEFINITIONS[experiment_name]
    defaults = [
        str(item["id"])
        for item in definition["configurations"]
        if bool(item["is_production_default"])
    ]
    if len(defaults) != 1:
        raise RuntimeError(f"Expected one production default for {experiment_name}")
    return defaults[0]


def _audit_note() -> str:
    hashes: list[str] = []
    for definition in EXPERIMENT_DEFINITIONS.values():
        logs_dir = ROOT / definition["directory"] / "logs"
        snapshots = sorted(logs_dir.glob("run_snapshot_*.json"), key=lambda path: path.stat().st_mtime)
        if not snapshots:
            continue
        import json

        payload = json.loads(snapshots[-1].read_text(encoding="utf-8"))
        value = str(payload.get("implementation_hash") or "")
        if value:
            hashes.append(value)
    unique = list(dict.fromkeys(hashes))
    if len(unique) <= 1:
        return "Formal Main8 held-out results (Dev10 excluded)."
    compact = " / ".join(value[:8] for value in unique)
    return (
        "PRELIMINARY — mixed production implementation fingerprints\n"
        f"({compact}).\n"
        "Rerun under a frozen implementation before publication."
    )


def _save_pair(figure: plt.Figure, output_dir: Path, stem: str, dpi: int, *, title: str) -> tuple[Path, Path]:
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, metadata={"Title": title}, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    return pdf_path, png_path


def _unavailable(axis: plt.Axes, title: str, audit: Mapping[str, Any]) -> None:
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_axis_off()
    axis.text(
        0.5,
        0.53,
        "Formal results incomplete",
        ha="center",
        va="center",
        fontweight="bold",
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.42,
        f"{audit['observed']}/{audit['expected']} configurations",
        ha="center",
        va="center",
        color="0.35",
        transform=axis.transAxes,
    )


def _line_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    *,
    title: str,
    x_field: str,
    xlabel: str,
    default_config_id: str | None = None,
) -> None:
    if not rows:
        _unavailable(axis, title, audit)
        return
    ordered = sorted(rows, key=lambda row: float(row[x_field]))
    precision_values = [float(row["precision"]) for row in ordered]
    recall_values = [float(row["recall"]) for row in ordered]
    x = [row[x_field] for row in ordered]
    f1_values = [float(row["f1"]) for row in ordered]
    axis.plot(x, f1_values, marker="o", linewidth=1.7, label="F1", color=F1_COLOR)
    axis.plot(
        x,
        precision_values,
        marker="s",
        linewidth=1.7,
        label="Precision",
        color=PRECISION_COLOR,
    )
    axis.plot(
        x,
        recall_values,
        marker="^",
        linewidth=1.7,
        label="Recall",
        color=RECALL_COLOR,
    )
    if default_config_id is not None:
        defaults = [row for row in ordered if row["config_id"] == default_config_id]
        if len(defaults) == 1:
            default = defaults[0]
            default_x = default[x_field]
            axis.axvline(default_x, color=DEFAULT_COLOR, linestyle="--", linewidth=0.9, alpha=0.75)
            axis.scatter(
                [default_x, default_x, default_x],
                [default["f1"], default["precision"], default["recall"]],
                s=100,
                marker="*",
                color=DEFAULT_COLOR,
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
                label="Production default",
            )
    axis.set_xticks(x)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Category-balanced score")
    _format_score_axis(axis, [*f1_values, *precision_values, *recall_values])
    axis.set_title(title, loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=2)


def _graph_panel(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]
) -> None:
    title = "(a) Graph–geometry coupling"
    if not rows:
        _unavailable(axis, title, audit)
        return
    positions = np.arange(len(rows))
    precision_values = [float(row["precision"]) for row in rows]
    recall_values = [float(row["recall"]) for row in rows]
    f1_values = [float(row["f1"]) for row in rows]
    axis.plot(
        positions,
        f1_values,
        marker="o",
        markersize=5.8,
        linewidth=1.9,
        label="F1",
        color=F1_COLOR,
    )
    axis.plot(
        positions,
        precision_values,
        marker="s",
        markersize=5.4,
        linewidth=1.9,
        label="Precision",
        color=PRECISION_COLOR,
    )
    axis.plot(
        positions,
        recall_values,
        marker="^",
        markersize=5.6,
        linewidth=1.9,
        label="Recall",
        color=RECALL_COLOR,
    )
    default_id = _default_id("graph_geometry_coupling")
    default_index = next(
        (index for index, row in enumerate(rows) if row["config_id"] == default_id),
        None,
    )
    if default_index is not None:
        axis.axvline(
            default_index,
            color=DEFAULT_COLOR,
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
        )
        axis.scatter(
            [default_index, default_index, default_index],
            [
                f1_values[default_index],
                precision_values[default_index],
                recall_values[default_index],
            ],
            s=140,
            marker="*",
            color=DEFAULT_COLOR,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
            label="Production default",
        )
    for position, value in zip(positions, recall_values):
        axis.annotate(
            f"{100.0 * value:.2f}%",
            (position, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    for position, value in zip(positions, precision_values):
        axis.annotate(
            f"{100.0 * value:.2f}%",
            (position, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    for position, value in zip(positions, f1_values):
        axis.annotate(
            f"{100.0 * value:.2f}%",
            (position, value),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.5,
        )
    design_labels = {
        "small_k8_L8-16-32": "small",
        "small_k10_L8-16-32": "compact scale",
        "local_k14_L12-24-48": "local -2",
        "default_k16_L12-24-48": "default",
        "local_k18_L12-24-48": "local +2",
        "scale_k24_L18-36-72": "1.5x scale",
        "large_k32_L24-48-96": "2x scale",
    }
    labels = [
        f"{design_labels.get(str(row['config_id']), str(row['config_id']))}"
        f"\nk={row['graph_knn_k']}"
        f"\nL={{{','.join(str(v) for v in row['surface_variation_scales'])}}}"
        for row in rows
    ]
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Category-balanced score")
    axis.margins(x=0.05)
    _format_score_axis(axis, [*f1_values, *precision_values, *recall_values])
    axis.set_title(title, loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=4)


def _candidate_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    figure: plt.Figure,
) -> None:
    title = "(d) Candidate recovery"
    if not rows:
        _unavailable(axis, title, audit)
        return
    tau_values = (0.30, 0.40, 0.50, 0.60, 0.70)
    hops_values = (5, 9, 13)
    matrix = np.full((len(tau_values), len(hops_values)), np.nan)
    tau_index = {value: index for index, value in enumerate(tau_values)}
    hops_index = {value: index for index, value in enumerate(hops_values)}
    for row in rows:
        matrix[tau_index[round(float(row["tau_q"]), 2)], hops_index[int(row["h_max"])]] = row["f1"]
    finite = matrix[np.isfinite(matrix)]
    color_padding = max(0.0015, 0.08 * float(np.ptp(finite))) if finite.size else 0.01
    vmin = max(0.0, float(np.min(finite)) - color_padding) if finite.size else 0.0
    vmax = min(1.0, float(np.max(finite)) + color_padding) if finite.size else 1.0
    mesh = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=BLUE_CMAP,
    )
    axis.set_xticks(range(len(hops_values)), hops_values)
    axis.set_yticks(range(len(tau_values)), [f"{value:.2f}" for value in tau_values])
    axis.set_xlabel(r"Maximum growth hops $H_{max}$")
    axis.set_ylabel(r"Seed threshold $\tau_Q$")
    axis.set_title(title, loc="left", fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "NA" if np.isnan(value) else f"{100.0 * value:.2f}"
            color = "black"
            axis.text(column_index, row_index, label, ha="center", va="center", color=color, fontsize=7)
    axis.add_patch(Rectangle((0.5, 1.5), 1.0, 1.0, fill=False, edgecolor=DEFAULT_COLOR, linewidth=2.0))
    colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Category-balanced F1")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))


def _candidate_metric_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    figure: plt.Figure,
) -> None:
    tau_values = (0.30, 0.40, 0.50, 0.60, 0.70)
    hops_values = (5, 9, 13)
    matrix = np.full((len(tau_values), len(hops_values)), np.nan)
    for row in rows:
        row_index = tau_values.index(round(float(row["tau_q"]), 2))
        column_index = hops_values.index(int(row["h_max"]))
        matrix[row_index, column_index] = float(row[metric])
    finite = matrix[np.isfinite(matrix)]
    padding = max(0.0015, 0.08 * float(np.ptp(finite)))
    vmin, vmax = float(np.min(finite)) - padding, float(np.max(finite)) + padding
    mesh = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=BLUE_CMAP,
    )
    axis.set_xticks(range(len(hops_values)), hops_values)
    axis.set_yticks(range(len(tau_values)), [f"{value:.2f}" for value in tau_values])
    axis.set_xlabel(r"Maximum growth hops $H_{max}$")
    axis.set_ylabel(r"Seed threshold $\tau_Q$")
    axis.set_title(title, loc="left", fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{100.0 * value:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7.5,
            )
    axis.add_patch(Rectangle((0.5, 1.5), 1.0, 1.0, fill=False, edgecolor=DEFAULT_COLOR, linewidth=2.0))
    colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))


def _write_plot_data(
    output_dir: Path,
    experiment_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    definition = EXPERIMENT_DEFINITIONS[experiment_name]
    path = output_dir / "data" / f"{definition['directory']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config_id",
        *definition["parameter_columns"],
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            if "surface_variation_scales" in row:
                row["surface_variation_scales"] = "[" + ",".join(
                    str(value) for value in row["surface_variation_scales"]
                ) + "]"
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dpi < 300:
        raise ValueError("--dpi must be at least 300")
    names = tuple(EXPERIMENT_DEFINITIONS)
    datasets: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for name in names:
        datasets[name], audits[name] = _read_summary(name, scope=args.scope)
    incomplete = [name for name in names if not audits[name]["complete"]]
    if incomplete and not args.allow_incomplete:
        details = "; ".join(f"{name}: {audits[name]['missing']}" for name in incomplete)
        raise RuntimeError(
            "Formal 37-configuration sensitivity results are incomplete; "
            f"paper figure was not generated. {details}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        _write_plot_data(output_dir, name, datasets[name])
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    _graph_panel(axes[0, 0], datasets["graph_geometry_coupling"], audits["graph_geometry_coupling"])
    _line_panel(
        axes[0, 1],
        datasets["vahks_modulation"],
        audits["vahks_modulation"],
        title="(b) VA-HKS modulation",
        x_field="gamma",
        xlabel=r"Modulation coefficient $\gamma$",
        default_config_id=_default_id("vahks_modulation"),
    )
    _line_panel(
        axes[1, 0],
        datasets["ac_boundary"],
        audits["ac_boundary"],
        title="(c) New-AC boundary",
        x_field="theta_ac_deg",
        xlabel=r"Angular threshold $\theta_{AC}$ (degree)",
        default_config_id=_default_id("ac_boundary"),
    )
    _candidate_panel(
        axes[1, 1],
        datasets["candidate_recovery"],
        audits["candidate_recovery"],
        figure,
    )
    figure.suptitle(f"COVERT parameter sensitivity ({args.scope})", fontweight="bold")
    figure.text(0.5, -0.025, _audit_note(), ha="center", va="top", fontsize=6.5, color="0.35")
    main_pdf = output_dir / "covert_sensitivity_main.pdf"
    main_png = output_dir / "covert_sensitivity_main.png"
    figure.savefig(main_pdf, metadata={"Title": "COVERT parameter sensitivity"})
    figure.savefig(main_png, dpi=args.dpi, facecolor="white")
    plt.close(figure)

    supplementary, axis = plt.subplots(1, 1, figsize=(4.5, 3.4), constrained_layout=True)
    _line_panel(
        axis,
        datasets["gtr_local_calibration"],
        audits["gtr_local_calibration"],
        title="GTR local-normal calibration",
        x_field="tau_rel",
        xlabel=r"Relative normal-response threshold $\tau_{rel}$",
        default_config_id=_default_id("gtr_local_calibration"),
    )
    supplementary_pdf = output_dir / "covert_sensitivity_gtr_supplementary.pdf"
    supplementary_png = output_dir / "covert_sensitivity_gtr_supplementary.png"
    supplementary.savefig(supplementary_pdf, metadata={"Title": "COVERT GTR sensitivity"})
    supplementary.savefig(supplementary_png, dpi=args.dpi, facecolor="white")
    plt.close(supplementary)

    individual_outputs: list[Path] = []

    graph_figure, graph_axis = plt.subplots(1, 1, figsize=(5.4, 3.8), constrained_layout=True)
    _graph_panel(graph_axis, datasets["graph_geometry_coupling"], audits["graph_geometry_coupling"])
    graph_axis.set_title("Graph–geometry coupling", loc="left", fontweight="bold")
    graph_figure.text(0.5, -0.025, _audit_note(), ha="center", va="top", fontsize=6.2, color="0.35")
    individual_outputs.extend(
        _save_pair(
            graph_figure,
            output_dir,
            "01_graph_geometry_coupling",
            args.dpi,
            title="COVERT graph–geometry coupling sensitivity",
        )
    )
    plt.close(graph_figure)

    line_specs = (
        (
            "vahks_modulation",
            "gamma",
            r"Modulation coefficient $\gamma$",
            "VA-HKS modulation",
            "02_vahks_modulation",
        ),
        (
            "ac_boundary",
            "theta_ac_deg",
            r"Angular threshold $\theta_{AC}$ (degree)",
            "New-AC boundary",
            "03_ac_boundary",
        ),
        (
            "gtr_local_calibration",
            "tau_rel",
            r"Relative normal-response threshold $\tau_{rel}$",
            "GTR local-normal calibration",
            "05_gtr_local_calibration",
        ),
    )
    for experiment_name, x_field, xlabel, title, stem in line_specs:
        item_figure, item_axis = plt.subplots(1, 1, figsize=(5.4, 3.8), constrained_layout=True)
        _line_panel(
            item_axis,
            datasets[experiment_name],
            audits[experiment_name],
            title=title,
            x_field=x_field,
            xlabel=xlabel,
            default_config_id=_default_id(experiment_name),
        )
        item_figure.text(0.5, -0.025, _audit_note(), ha="center", va="top", fontsize=6.2, color="0.35")
        individual_outputs.extend(
            _save_pair(item_figure, output_dir, stem, args.dpi, title=f"COVERT {title} sensitivity")
        )
        plt.close(item_figure)

    candidate_figure, candidate_axis = plt.subplots(1, 1, figsize=(5.4, 3.9), constrained_layout=True)
    _candidate_panel(
        candidate_axis,
        datasets["candidate_recovery"],
        audits["candidate_recovery"],
        figure=candidate_figure,
    )
    candidate_axis.set_title("Candidate recovery sensitivity", loc="left", fontweight="bold")
    candidate_figure.text(0.5, -0.025, _audit_note(), ha="center", va="top", fontsize=6.2, color="0.35")
    individual_outputs.extend(
        _save_pair(
            candidate_figure,
            output_dir,
            "04_candidate_recovery",
            args.dpi,
            title="COVERT candidate recovery sensitivity",
        )
    )
    plt.close(candidate_figure)

    overview, overview_axes = plt.subplots(3, 2, figsize=(10.0, 10.0), constrained_layout=True)
    _graph_panel(
        overview_axes[0, 0],
        datasets["graph_geometry_coupling"],
        audits["graph_geometry_coupling"],
    )
    _line_panel(
        overview_axes[0, 1],
        datasets["vahks_modulation"],
        audits["vahks_modulation"],
        title="(b) VA-HKS modulation",
        x_field="gamma",
        xlabel=r"Modulation coefficient $\gamma$",
        default_config_id=_default_id("vahks_modulation"),
    )
    _line_panel(
        overview_axes[1, 0],
        datasets["ac_boundary"],
        audits["ac_boundary"],
        title="(c) New-AC boundary",
        x_field="theta_ac_deg",
        xlabel=r"Angular threshold $\theta_{AC}$ (degree)",
        default_config_id=_default_id("ac_boundary"),
    )
    _candidate_panel(
        overview_axes[1, 1],
        datasets["candidate_recovery"],
        audits["candidate_recovery"],
        overview,
    )
    _line_panel(
        overview_axes[2, 0],
        datasets["gtr_local_calibration"],
        audits["gtr_local_calibration"],
        title="(e) GTR local-normal calibration",
        x_field="tau_rel",
        xlabel=r"Relative normal-response threshold $\tau_{rel}$",
        default_config_id=_default_id("gtr_local_calibration"),
    )
    overview_axes[2, 1].set_axis_off()
    overview_axes[2, 1].text(
        0.02,
        0.92,
        "Reading guide",
        transform=overview_axes[2, 1].transAxes,
        fontweight="bold",
        fontsize=10,
    )
    overview_axes[2, 1].text(
        0.02,
        0.77,
        "Blue: category-balanced F1\nRed: category-balanced Precision\nGreen: category-balanced Recall\nDark-red star / outline: production default\nF1 heatmap runs from light blue (low) to deep blue (high)",
        transform=overview_axes[2, 1].transAxes,
        va="top",
        linespacing=1.55,
        fontsize=8.5,
    )
    overview.suptitle("COVERT parameter sensitivity — held-out Main8", fontweight="bold")
    overview.text(0.5, -0.012, _audit_note(), ha="center", va="top", fontsize=6.5, color="0.35")
    overview_pdf, overview_png = _save_pair(
        overview,
        output_dir,
        "covert_sensitivity_overview",
        args.dpi,
        title="COVERT parameter sensitivity overview",
    )
    plt.close(overview)
    print(f"scope={args.scope}")
    print(f"main_pdf={main_pdf}")
    print(f"main_png={main_png}")
    print(f"supplementary_pdf={supplementary_pdf}")
    print(f"supplementary_png={supplementary_png}")
    print(f"overview_pdf={overview_pdf}")
    print(f"overview_png={overview_png}")
    for path in individual_outputs:
        print(f"individual={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
