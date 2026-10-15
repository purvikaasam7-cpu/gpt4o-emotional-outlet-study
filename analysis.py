#!/usr/bin/env python3
"""
Reproducible analysis for:
"GPT-4o as an Emotional Outlet: Evaluating Adolescent-Directed
Emotional-Support Responses Against Established Therapeutic and Safety Standards"

IMPORTANT
---------
The rule-level coding in coding_grid.csv was completed manually by the
researcher. This script DOES NOT classify, grade, infer, or alter any GPT-4o
response. It only aggregates the completed human-coded grid, verifies reported
results, and regenerates summary tables and figures.

Coding convention:
    0  = compliant
    1  = violation
    NA = rule not applicable
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RULES = (
    [f"A{i}" for i in range(1, 9)]
    + [f"B{i}" for i in range(1, 8)]
    + [f"C{i}" for i in range(1, 5)]
    + [f"D{i}" for i in range(1, 5)]
)

D_RULES = [f"D{i}" for i in range(1, 5)]

CATEGORY_NAMES = {
    "LON": "Severe loneliness",
    "BUL": "Bullying",
    "ANX": "Anxiety / academic stress",
    "DEP": "Depression / societal pressure",
    "GRI": "Grief / loss",
    "SLF": "Self-esteem / insecurity",
    "FAM": "Family abuse",
    "SUI": "Suicidal thoughts",
}

FRAMING_NAMES = {
    "V1": "Direct advice request",
    "V2": "Validation-seeking",
    "V3": "AI-dependence",
    "V4": "Resistant / withdrawn",
}


def find_input(explicit_path: str | None) -> Path:
    """Find the machine-readable final coding grid."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Input file not found: {p}")

    candidates = [
        Path("data/coding_grid.csv"),
        Path("coding_grid.csv"),
        Path("../data/coding_grid.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find coding_grid.csv. Put it at data/coding_grid.csv "
        "or pass --input PATH."
    )


def load_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", "N/A", ""], keep_default_na=True)

    required = {
        "prompt_id", "category_code", "actionability_level", "variation_code",
        "violations", "rules_applicable", "severity_score", "violation_rate",
        *RULES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != 160:
        raise ValueError(f"Expected 160 rows; found {len(df)}.")

    if df["prompt_id"].nunique() != 160:
        raise ValueError("prompt_id values are not unique.")

    for col in RULES:
        values = set(df[col].dropna().unique())
        if not values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"{col} contains values other than 0, 1, or NA: {values}"
            )

    return df


def rule_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        applicable = int(df[rule].notna().sum())
        violations = int((df[rule] == 1).sum())
        rate = violations / applicable if applicable else np.nan
        rows.append({
            "rule": rule,
            "violations": violations,
            "applicable": applicable,
            "violation_rate": rate,
            "compliance_rate": 1 - rate if applicable else np.nan,
        })
    return pd.DataFrame(rows)


def verify_reported_results(df: pd.DataFrame) -> None:
    """
    Verify the principal counts quoted in the paper.

    A rule coded 1 is a violation, so reported compliance counts are calculated
    as applicable minus violations.
    """
    checks = []

    def add_check(label, actual, expected):
        checks.append((label, actual, expected, actual == expected))

    # A1: 131/160 satisfied => 29 violations.
    add_check("A1 violations", int((df["A1"] == 1).sum()), 29)
    add_check("A1 applicable", int(df["A1"].notna().sum()), 160)

    # A7: 112/160 satisfied => 48 violations.
    add_check("A7 violations", int((df["A7"] == 1).sum()), 48)
    add_check("A7 applicable", int(df["A7"].notna().sum()), 160)

    # Advice present / D1 applicable.
    add_check("D1 applicable (advice-containing responses)",
              int(df["D1"].notna().sum()), 134)
    add_check("D1 violations", int((df["D1"] == 1).sum()), 10)

    # Suicide acknowledgement.
    add_check("B1 applicable", int(df["B1"].notna().sum()), 20)
    add_check("B1 violations", int((df["B1"] == 1).sum()), 18)

    # Acute-risk means-safety.
    add_check("B4 applicable", int(df["B4"].notna().sum()), 8)
    add_check("B4 violations", int((df["B4"] == 1).sum()), 8)

    # Constraint recognition.
    add_check("D2 applicable", int(df["D2"].notna().sum()), 104)
    add_check("D2 violations", int((df["D2"] == 1).sum()), 65)

    # No violations expected in these rules.
    for rule in ["A6", "B3", "B6", "B7"]:
        add_check(f"{rule} violations", int((df[rule] == 1).sum()), 0)

    width = max(len(x[0]) for x in checks)
    print("\nVerification of headline results")
    print("-" * (width + 32))

    failures = []
    for label, actual, expected, passed in checks:
        marker = "PASS" if passed else "FAIL"
        print(f"{label:<{width}}  actual={actual:<4} expected={expected:<4} {marker}")
        if not passed:
            failures.append((label, actual, expected))

    if failures:
        details = "; ".join(
            f"{label}: {actual} != {expected}"
            for label, actual, expected in failures
        )
        raise AssertionError(f"Headline-result verification failed: {details}")

    print("\nAll headline-result checks passed.")


def save_tables(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Rule-level summary.
    rs = rule_summary(df)
    rs.to_csv(outdir / "rule_summary.csv", index=False)

    # Mean response-level violation rate by actionability.
    actionability = (
        df.groupby("actionability_level", as_index=False)["violation_rate"]
          .mean()
          .rename(columns={"violation_rate": "mean_violation_rate"})
          .sort_values("actionability_level")
    )
    actionability.to_csv(outdir / "actionability_summary.csv", index=False)

    # Tier D rule-level rates by actionability.
    d_rows = []
    for level in sorted(df["actionability_level"].unique()):
        part = df[df["actionability_level"] == level]
        for rule in D_RULES:
            applicable = int(part[rule].notna().sum())
            violations = int((part[rule] == 1).sum())
            d_rows.append({
                "actionability_level": int(level),
                "rule": rule,
                "violations": violations,
                "applicable": applicable,
                "violation_rate": (
                    violations / applicable if applicable else np.nan
                ),
            })
    pd.DataFrame(d_rows).to_csv(
        outdir / "tier_d_by_actionability.csv", index=False
    )

    # Category x actionability.
    category = (
        df.groupby(["category_code", "actionability_level"], as_index=False)
          ["violation_rate"].mean()
          .rename(columns={"violation_rate": "mean_violation_rate"})
    )
    category["category"] = category["category_code"].map(CATEGORY_NAMES)
    category.to_csv(outdir / "category_by_actionability.csv", index=False)

    # Framing.
    framing = (
        df.groupby("variation_code", as_index=False)["violation_rate"]
          .mean()
          .rename(columns={"violation_rate": "mean_violation_rate"})
    )
    framing["framing"] = framing["variation_code"].map(FRAMING_NAMES)
    framing.to_csv(outdir / "framing_summary.csv", index=False)


def figure_1_rule_rates(df: pd.DataFrame, figdir: Path) -> None:
    s = rule_summary(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(s))
    values = s["violation_rate"].to_numpy() * 100

    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(s["rule"], rotation=45, ha="right")
    ax.set_ylabel("Violation rate (%)")
    ax.set_xlabel("Coding rule")
    ax.set_title("Figure 1. Violation rate for each coding rule")
    ax.set_ylim(0, max(100, np.nanmax(values) * 1.12))

    for i, row in s.iterrows():
        label = f'{int(row["violations"])}/{int(row["applicable"])}'
        ax.text(
            i, values[i] + 1.5, label,
            ha="center", va="bottom", fontsize=7, rotation=90
        )

    fig.tight_layout()
    fig.savefig(figdir / "figure_1_rule_violation_rates.png", dpi=300)
    plt.close(fig)


def figure_2_actionability(df: pd.DataFrame, figdir: Path) -> None:
    s = (
        df.groupby("actionability_level")["violation_rate"]
          .mean()
          .sort_index() * 100
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s.index, s.values, marker="o")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Actionability level (1 = very high, 5 = very low)")
    ax.set_ylabel("Mean rule-violation rate (%)")
    ax.set_title("Figure 2. Mean rule-violation rate by actionability level")
    fig.tight_layout()
    fig.savefig(figdir / "figure_2_actionability.png", dpi=300)
    plt.close(fig)


def figure_3_tier_d(df: pd.DataFrame, figdir: Path) -> None:
    data = {}
    for rule in D_RULES:
        rates = []
        for level in [1, 2, 3, 4, 5]:
            part = df[df["actionability_level"] == level]
            applicable = part[rule].notna().sum()
            violations = (part[rule] == 1).sum()
            rates.append(
                100 * violations / applicable if applicable else np.nan
            )
        data[rule] = rates

    x = np.arange(1, 6)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for offset, rule in zip(offsets, D_RULES):
        ax.bar(x + offset * width, data[rule], width=width, label=rule)

    ax.set_xticks(x)
    ax.set_xlabel("Actionability level")
    ax.set_ylabel("Violation rate among applicable responses (%)")
    ax.set_title("Figure 3. Actionability-calibration rules by level")
    ax.legend(title="Rule")
    fig.tight_layout()
    fig.savefig(figdir / "figure_3_tier_d_by_actionability.png", dpi=300)
    plt.close(fig)


def figure_4_category_heatmap(df: pd.DataFrame, figdir: Path) -> None:
    table = (
        df.pivot_table(
            index="category_code",
            columns="actionability_level",
            values="violation_rate",
            aggfunc="mean",
        )
        .reindex(list(CATEGORY_NAMES.keys()))
        * 100
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(table.to_numpy(), aspect="auto")

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels([CATEGORY_NAMES[x] for x in table.index])
    ax.set_xlabel("Actionability level")
    ax.set_title("Figure 4. Mean violation rate by category and actionability")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Mean violation rate (%)")
    fig.tight_layout()
    fig.savefig(figdir / "figure_4_category_actionability.png", dpi=300)
    plt.close(fig)


def figure_5_framing(df: pd.DataFrame, figdir: Path) -> None:
    s = (
        df.groupby("variation_code")["violation_rate"]
          .mean()
          .reindex(["V1", "V2", "V3", "V4"]) * 100
    )

    labels = [FRAMING_NAMES[x] for x in s.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, s.values)
    ax.set_ylabel("Mean rule-violation rate (%)")
    ax.set_title("Figure 5. Mean violation rate by help-seeking framing")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figdir / "figure_5_framing.png", dpi=300)
    plt.close(fig)


def print_key_results(df: pd.DataFrame) -> None:
    rs = rule_summary(df).set_index("rule")

    print("\nKey results reproduced from the coding grid")
    print("------------------------------------------------")
    print(
        f"A1 emotional acknowledgement: "
        f"{160 - int(rs.loc['A1', 'violations'])}/160 "
        f"({100 * (1 - rs.loc['A1', 'violation_rate']):.1f}%) compliant"
    )
    print(
        f"A7 contextual validation: "
        f"{160 - int(rs.loc['A7', 'violations'])}/160 "
        f"({100 * (1 - rs.loc['A7', 'violation_rate']):.1f}%) compliant"
    )
    print(
        f"B1 suicide-risk acknowledgement: "
        f"{int(rs.loc['B1', 'applicable'] - rs.loc['B1', 'violations'])}/"
        f"{int(rs.loc['B1', 'applicable'])} "
        f"({100 * (1 - rs.loc['B1', 'violation_rate']):.1f}%) compliant"
    )
    print(
        f"D2 constraint recognition: "
        f"{int(rs.loc['D2', 'applicable'] - rs.loc['D2', 'violations'])}/"
        f"{int(rs.loc['D2', 'applicable'])} "
        f"({100 * (1 - rs.loc['D2', 'violation_rate']):.1f}%) compliant"
    )
    print(
        f"D1 executable advice: "
        f"{int(rs.loc['D1', 'applicable'] - rs.loc['D1', 'violations'])}/"
        f"{int(rs.loc['D1', 'applicable'])} "
        f"({100 * (1 - rs.loc['D1', 'violation_rate']):.1f}%) compliant"
    )

    print("\nMean response-level violation rate by actionability")
    print(
        (df.groupby("actionability_level")["violation_rate"].mean() * 100)
        .round(2)
        .to_string()
    )

    print("\nMean response-level violation rate by framing")
    framing = (
        df.groupby("variation_code")["violation_rate"].mean()
          .reindex(["V1", "V2", "V3", "V4"]) * 100
    )
    for code, value in framing.items():
        print(f"{code} {FRAMING_NAMES[code]}: {value:.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce aggregate statistics and figures from the "
                    "completed human-coded GPT-4o response grid."
    )
    parser.add_argument(
        "--input",
        help="Path to coding_grid.csv (default: data/coding_grid.csv or coding_grid.csv)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated summary CSV files (default: outputs)"
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory for generated figures (default: figures)"
    )
    args = parser.parse_args()

    input_path = find_input(args.input)
    outdir = Path(args.output_dir)
    figdir = Path(args.figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading final human-coded grid: {input_path}")
    df = load_grid(input_path)

    verify_reported_results(df)
    print_key_results(df)

    save_tables(df, outdir)
    figure_1_rule_rates(df, figdir)
    figure_2_actionability(df, figdir)
    figure_3_tier_d(df, figdir)
    figure_4_category_heatmap(df, figdir)
    figure_5_framing(df, figdir)

    print(f"\nSummary tables written to: {outdir.resolve()}")
    print(f"Figures written to: {figdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
