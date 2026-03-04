import matplotlib.pyplot as plt
import numpy as np
import pickle
import colorsys

# --- Load the results --- #
with open(
    "/home/charbie/Documents/Programmation/SOCP/results/Vertebrate_VariationalPolynomial_MeanAndCovariance_CVG_1p0e-08_2026-03-03-16-33_.pkl",
    "rb",
) as f:
    data_VariationalPolynomial_MeanAndCovariance = pickle.load(f)
    n_shooting = data_VariationalPolynomial_MeanAndCovariance["difference_between_means"].shape[0] - 1

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/Vertebrate_VariationalPolynomial_NoiseDiscretization_CVG_1p0e-08_2026-03-03-16-32_.pkl",
    "rb",
) as f:
    data_VariationalPolynomial_Noise = pickle.load(f)

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/Vertebrate_Variational_NoiseDiscretization_CVG_1p0e-08_2026-03-03-16-09_.pkl",
    "rb",
) as f:
    data_Variational_Noise = pickle.load(f)

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/Vertebrate_DirectCollocationTrapezoidal_MeanAndCovariance_CVG_1p0e-08_2026-03-03-16-05_.pkl",
    "rb",
) as f:
    data_Trapezoidal_MeanAndCovariance = pickle.load(f)

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/Vertebrate_DirectCollocationTrapezoidal_NoiseDiscretization_CVG_1p0e-08_2026-03-03-16-03_.pkl",
    "rb",
) as f:
    data_Trapezoidal_Noise = pickle.load(f)

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/Vertebrate_DirectMultipleShooting_MeanAndCovariance_CVG_1p0e-08_2026-03-03-16-03_.pkl",
    "rb",
) as f:
    data_DirectMultipleShooting_MeanAndCovariance = pickle.load(f)

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/Vertebrate_DirectMultipleShooting_NoiseDiscretization_CVG_1p0e-08_2026-03-03-16-00_.pkl",
    "rb",
) as f:
    data_DirectMultipleShooting_Noise = pickle.load(f)

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/Vertebrate_DirectCollocationPolynomial_MeanAndCovariance_CVG_1p0e-08_2026-03-03-15-57_.pkl",
    "rb",
) as f:
    data_DirectCollocationPolynomial_MeanAndCovariance = pickle.load(f)

with open(
    "//home/charbie/Documents/Programmation/SOCP/results/Vertebrate_DirectCollocationPolynomial_NoiseDiscretization_CVG_1p0e-08_2026-03-03-15-11_.pkl",
    "rb",
) as f:
    data_DirectCollocationPolynomial_Noise = pickle.load(f)


# # --- Plot the comparison metrics --- #
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#
# axs[0].plot(np.linspace(0, n_shooting, n_shooting + 1), np.zeros((n_shooting + 1,)), "-k")
# axs[0].plot(data_DirectCollocationPolynomial_Noise["norm_difference_between_means"], "--", color="tab:red", label="Direct Collocation Polynomial x Noise Sampling")
# axs[0].plot(data_Trapezoidal_Noise["norm_difference_between_means"], "--", color="tab:green", label="Direct Collocation Trapezoidal x Noise Sampling")
# axs[0].plot(data_DirectMultipleShooting_Noise["norm_difference_between_means"], "--", color="tab:orange", label="Direct Multiple Shooting x Noise Sampling")
# axs[0].plot(data_Variational_Noise["norm_difference_between_means"], "--", color="tab:blue", label="Variational Trapezoidal x Noise Sampling")
# axs[0].plot(data_VariationalPolynomial_Noise["norm_difference_between_means"], "--", color="tab:purple", label="Variational Polynomial x Noise Sampling")
# axs[0].plot(data_DirectCollocationPolynomial_MeanAndCovariance["norm_difference_between_means"], ":", color="tab:red", label="Direct Collocation Polynomial x Mean and Covariance")
# axs[0].plot(data_Trapezoidal_MeanAndCovariance["norm_difference_between_means"], ":", color="tab:green", label="Direct Collocation Trapezoidal x Mean and Covariance")
# axs[0].plot(data_DirectMultipleShooting_MeanAndCovariance["norm_difference_between_means"], ":", color="tab:orange", label="Direct Multiple Shooting x Mean and Covariance")
# # axs[0].plot(data_DirectCollocationPolynomial_MeanAndCovariance["norm_difference_between_means"], ":", color="tab:blue", label="DirectCollocationPolynomial x Mean and Covariance")
# axs[0].plot(data_VariationalPolynomial_MeanAndCovariance["norm_difference_between_means"], ":", color="tab:purple", label="Variational Polynomial x Mean and Covariance")
# axs[0].set_title(r"$||\bar{q}_{opt} - \bar{q}_{sim}||$")
# axs[0].set_xlabel("Shooting node")
# axs[0].set_ylabel("Difference")
# axs[0].set_yscale("log")
#
# axs[1].plot(np.linspace(0, n_shooting, n_shooting + 1), np.zeros((n_shooting + 1)), "-k")
# axs[1].plot(data_DirectCollocationPolynomial_Noise["norm_difference_between_covs"], "--", color="tab:red", label="Direct Collocation Polynomial x Noise Sampling")
# axs[1].plot(data_Trapezoidal_Noise["norm_difference_between_covs"], "--", color="tab:green", label="Direct Collocation Trapezoidal x Noise Sampling")
# axs[1].plot(data_DirectMultipleShooting_Noise["norm_difference_between_covs"], "--", color="tab:orange", label="Direct Multiple Shooting x Noise Sampling")
# axs[1].plot(data_Variational_Noise["norm_difference_between_covs"], "--", color="tab:blue", label="Variational Trapezoidal x Noise Sampling")
# axs[1].plot(data_VariationalPolynomial_Noise["norm_difference_between_covs"], "--", color="tab:purple", label="Variational Polynomial x Noise Sampling")
# axs[1].plot(data_DirectCollocationPolynomial_MeanAndCovariance["norm_difference_between_covs"], ":", color="tab:red", label="Direct Collocation Polynomial x Mean and Covariance")
# axs[1].plot(data_Trapezoidal_MeanAndCovariance["norm_difference_between_covs"], ":", color="tab:green", label="Direct Collocation Trapezoidal x Mean and Covariance")
# axs[1].plot(data_DirectMultipleShooting_MeanAndCovariance["norm_difference_between_covs"], ":", color="tab:orange", label="Direct Multiple Shooting x Mean and Covariance")
# # axs[1].plot(data_DirectCollocationPolynomial_MeanAndCovariance["norm_difference_between_covs"], ":", color="tab:blue", label="DirectCollocationPolynomial x Mean and Covariance")
# axs[1].plot(data_VariationalPolynomial_MeanAndCovariance["norm_difference_between_covs"], ":", color="tab:purple", label="Variational Polynomial x Mean and Covariance")
# axs[1].set_title(r"$||P_{opt} - P_{sim}||_{2}$")
# axs[1].set_xlabel("Shooting node")
# axs[1].set_yscale("log")
#
# # axs[2].bar(0, data_DC_MAC["computation_time"], width=0.4, color="tab:red", label="DC & Mean+COV (Gillis)")
# # axs[2].bar(0.5, data_DMS_N["computation_time"], width=0.4, color="tab:purple", label="DMS & Noise")
# # axs[2].set_xlabel("Computation Time [s]")
# #
# # axs[3].bar(0, data_DC_MAC["optimal_cost"], width=0.4, color="tab:red", alpha=0.5, label="DC Simulation")
# # axs[3].bar(0.5, data_DMS_N["optimal_cost"], width=0.4, color="tab:purple", alpha=0.5, label="DMS Simulation")
# # axs[3].set_xlabel("Optimal Cost")
#
# axs[0].legend(bbox_to_anchor=(1.1, -0.15), loc="upper center", ncol=2)
# plt.subplots_adjust(bottom=0.3)
# # plt.tight_layout()
# plt.savefig("results/vertebrate_analysis.png", dpi=300)
# plt.show()


# --- Create the LaTeX result table --- #
DATA = {
    "DCP": {
        "NS": {
            "nb var": data_DirectCollocationPolynomial_Noise["nb_variables"],
            "nb const": data_DirectCollocationPolynomial_Noise["nb_constraints"],
            "time": data_DirectCollocationPolynomial_Noise["computational_time"],
            "nb inter": data_DirectCollocationPolynomial_Noise["nb_iterations"],
            "cost": data_DirectCollocationPolynomial_Noise["optimal_cost"],
        },
        "MaC": {
            "nb var": data_DirectCollocationPolynomial_MeanAndCovariance["nb_variables"],
            "nb const": data_DirectCollocationPolynomial_MeanAndCovariance["nb_constraints"],
            "time": data_DirectCollocationPolynomial_MeanAndCovariance["computational_time"],
            "nb inter": data_DirectCollocationPolynomial_MeanAndCovariance["nb_iterations"],
            "cost": data_DirectCollocationPolynomial_MeanAndCovariance["optimal_cost"],
        },
    },
    "DCT": {
        "NS": {
            "nb var": data_Trapezoidal_Noise["nb_variables"],
            "nb const": data_Trapezoidal_Noise["nb_constraints"],
            "time": data_Trapezoidal_Noise["computational_time"],
            "nb inter": data_Trapezoidal_Noise["nb_iterations"],
            "cost": data_Trapezoidal_Noise["optimal_cost"],
        },
        "MaC": {
            "nb var": data_Trapezoidal_MeanAndCovariance["nb_variables"],
            "nb const": data_Trapezoidal_MeanAndCovariance["nb_constraints"],
            "time": data_Trapezoidal_MeanAndCovariance["computational_time"],
            "nb inter": data_Trapezoidal_MeanAndCovariance["nb_iterations"],
            "cost": data_Trapezoidal_MeanAndCovariance["optimal_cost"],
        },
    },
    "DMS": {
        "NS": {
            "nb var": data_DirectMultipleShooting_Noise["nb_variables"],
            "nb const": data_DirectMultipleShooting_Noise["nb_constraints"],
            "time": data_DirectMultipleShooting_Noise["computational_time"],
            "nb inter": data_DirectMultipleShooting_Noise["nb_iterations"],
            "cost": data_DirectMultipleShooting_Noise["optimal_cost"],
        },
        "MaC": {
            "nb var": data_DirectMultipleShooting_MeanAndCovariance["nb_variables"],
            "nb const": data_DirectMultipleShooting_MeanAndCovariance["nb_constraints"],
            "time": data_DirectMultipleShooting_MeanAndCovariance["computational_time"],
            "nb inter": data_DirectMultipleShooting_MeanAndCovariance["nb_iterations"],
            "cost": data_DirectMultipleShooting_MeanAndCovariance["optimal_cost"],
        },
    },
    "TDMaOC": {
        "NS": {
            "nb var": data_Variational_Noise["nb_variables"],
            "nb const": data_Variational_Noise["nb_constraints"],
            "time": data_Variational_Noise["computational_time"],
            "nb inter": data_Variational_Noise["nb_iterations"],
            "cost": data_Variational_Noise["optimal_cost"],
        },
        "MaC": {
            "nb var": None,
            "nb const": None,
            "time": None,
            "nb inter": None,
            "cost": None,
        },
    },
    "PMCaOC": {
        "NS": {
            "nb var": data_VariationalPolynomial_Noise["nb_variables"],
            "nb const": data_VariationalPolynomial_Noise["nb_constraints"],
            "time": data_VariationalPolynomial_Noise["computational_time"],
            "nb inter": data_VariationalPolynomial_Noise["nb_iterations"],
            "cost": data_VariationalPolynomial_Noise["optimal_cost"],
        },
        "MaC": {
            "nb var": data_VariationalPolynomial_MeanAndCovariance["nb_variables"],
            "nb const": data_VariationalPolynomial_MeanAndCovariance["nb_constraints"],
            "time": data_VariationalPolynomial_MeanAndCovariance["computational_time"],
            "nb inter": data_VariationalPolynomial_MeanAndCovariance["nb_iterations"],
            "cost": data_VariationalPolynomial_MeanAndCovariance["optimal_cost"],
        },
    },
}

# Column order for the numeric metrics
METRIC_COLS = ["nb var", "nb const", "time", "nb inter", "cost"]

# Column headers
METRIC_HEADERS = [r"\# var.", r"\# const.", "Time [s]", r"\# iter.", "Cost"]


# ── Color helpers ─────────────────────────────────────────────────────────────


def value_to_rgb(value: float, vmin: float, vmax: float):
    """Map value in [vmin, vmax] → RGB (red = low, green = high)."""
    t = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    hue = t * 120 / 360  # 0° red → 120° green
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
    return int(r * 255), int(g * 255), int(b * 255)


# ── Flatten data & compute per-metric min/max ─────────────────────────────────

flat_rows = []  # (trans, title, metrics)
for trans, titles in DATA.items():
    for title, metrics in titles.items():
        flat_rows.append((trans, title, metrics))

col_values = {m: [] for m in METRIC_COLS}
for _, _, metrics in flat_rows:
    for m in METRIC_COLS:
        if metrics.get(m) is not None:
            col_values[m].append(float(metrics[m]))
        else:
            col_values[m].append(np.nan)

col_min = {m: min(v) for m, v in col_values.items()}
col_max = {m: max(v) for m, v in col_values.items()}


# ── Build color definitions and table rows ────────────────────────────────────

color_defs = []
table_rows = []

trans_seen = {}

for row_idx, (trans, title, metrics) in enumerate(flat_rows):
    span = sum(1 for t, _, _ in flat_rows if t == trans)
    cells = []

    # Transcription cell (multirow on first occurrence)
    if trans not in trans_seen:
        cells.append(rf"\multirow{{{span}}}{{*}}{{{trans}}}")
        trans_seen[trans] = True
    else:
        cells.append("")

    # Title cell
    cells.append(title)

    # Numeric / colored cells
    for col_idx, metric in enumerate(METRIC_COLS):
        val = metrics.get(metric)
        if val is not None:
            fval = float(val)
            r, g, b = value_to_rgb(fval, col_min[metric], col_max[metric])
            cname = f"cell{row_idx}m{col_idx}"
            color_defs.append(rf"\definecolor{{{cname}}}{{RGB}}{{{r},{g},{b}}}")
            display = f"{fval:.2f}" if isinstance(val, float) else str(int(val))
            cells.append(rf"\cellcolor{{{cname}}}{display}")
        else:
            cells.append("")

    # Draw \hline only after last row of each transcription group
    is_last_in_group = row_idx == len(flat_rows) - 1 or flat_rows[row_idx + 1][0] != trans
    hline = r" \hline" if is_last_in_group else ""
    table_rows.append("    " + " & ".join(cells) + rf" \\{hline}")


# ── Assemble full LaTeX document ──────────────────────────────────────────────

col_spec = "|c|l|" + "c|" * len(METRIC_COLS)
color_block = "\n".join(color_defs)

header_cells = [
    r"\textbf{Trans.}",
    r"\textbf{Noise}",
] + [rf"\textbf{{{h}}}" for h in METRIC_HEADERS]

header_row = "    " + " & ".join(header_cells) + r" \\ \hline"

latex = (
    r"\documentclass{article}" + "\n"
    r"\usepackage[table]{xcolor}" + "\n"
    r"\usepackage{multirow}" + "\n"
    r"\usepackage{array}" + "\n"
    r"\usepackage{booktabs}" + "\n"
    "\n"
    "% Auto-generated cell colours\n" + color_block + "\n"
    "\n"
    r"\begin{document}" + "\n"
    "\n"
    r"\begin{table}[ht]" + "\n"
    r"  \centering" + "\n"
    r"  \caption{Comparison of the efficiency of all implementations. The cells are color coded from red (undesirable) to green (desirable).}"
    + "\n"
    r"  \renewcommand{\arraystretch}{1.4}" + "\n"
    rf"  \begin{{tabular}}{{{col_spec}}}" + "\n"
    r"    \hline" + "\n" + header_row + "\n"
    r"    \hline" + "\n" + "\n".join(table_rows) + "\n"
    r"  \end{tabular}" + "\n"
    r"\end{table}" + "\n"
    "\n"
    r"\end{document}" + "\n"
)

OUTPUT_FILE = "table.tex"
with open(OUTPUT_FILE, "w") as fh:
    fh.write(latex)

print(f"LaTeX file written to: {OUTPUT_FILE}")
print()
print("Customise the DATA dict at the top of the script and re-run.")
print("Compile with:  pdflatex table.tex")
