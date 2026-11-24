"""Figures for the replay.  One colour per method family, fixed across figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .contingency import quarter_end  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e8e7e3"
REFERENCE = "#a3a29c"

# Method family -> colour.  The order is fixed; a method keeps its colour everywhere.
COLORS = {
    "prr": "#2a78d6",
    "ror": "#eb6834",
    "bcpnn": "#1baf7a",
    "gps": "#eda100",
    "hmm_prior": "#e87ba4",
    "hmm_bw": "#008300",
}
DATA_A, DATA_B = "#2a78d6", "#eb6834"  # for two-variant comparisons of the data itself


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9, "axes.labelcolor": INK_2,
        "text.color": INK, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _plain_log(ax):
    """Log axis with plain tick labels (0.5, 1, 2, 10) instead of powers of ten."""
    ax.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def quarter_dates(quarters):
    return pd.to_datetime([quarter_end(q) for q in quarters])


def _mark_action(axes, date, label):
    date = pd.Timestamp(date)
    for ax in axes:
        ax.axvline(date, color=INK, linewidth=1, zorder=1)
    axes[0].annotate(label, xy=(date, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(4, 2), textcoords="offset points", fontsize=8, color=INK,
                     va="bottom", ha="left")


def _alarm_dot(ax, dates, values, flagged, color):
    """Ringed dot on the first quarter a method's rule is met."""
    flagged = np.asarray(flagged, dtype=bool)
    if flagged.any():
        i = int(np.argmax(flagged))
        ax.plot([dates[i]], [values[i]], "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=5)


def _legend(ax, **kwargs):
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, **kwargs)


def case_trajectory(traj, title, action_date, action_label, path, min_cases=3):
    """Four stacked panels for one historical case over the replay window."""
    dates = quarter_dates(traj["quarter"])
    eligible = traj["cum_a"].to_numpy() >= min_cases
    on_market = traj["cum_n_drug"].to_numpy() > 0   # nothing to score before the first report

    def series(column):
        return np.where(on_market, traj[column].to_numpy(dtype=float), np.nan)

    fig, axes = plt.subplots(4, 1, figsize=(8.4, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.32})

    ax = axes[0]
    observed = traj["a"].where(traj["a"] > 0).to_numpy(dtype=float)
    expected = traj["expected"].where(traj["expected"] > 0).to_numpy(dtype=float)
    ax.plot(dates, observed, color=DATA_A, marker="o", ms=3.5, label="Observed")
    ax.plot(dates, expected, color=INK_2, label="Expected at background rate")
    _plain_log(ax)
    ax.set_title("Reports per quarter mentioning the drug and the event", pad=16)
    _legend(ax)

    ax = axes[1]
    prr_lo, eb05 = series("prr_lower"), series("eb05")
    ax.plot(dates, prr_lo, color=COLORS["prr"], label="PRR, lower 95% bound")
    ax.plot(dates, eb05, color=COLORS["gps"], label="GPS, EB05")
    ax.axhline(1, color=REFERENCE, linewidth=1)
    ax.axhline(2, color=REFERENCE, linewidth=1)
    _plain_log(ax)
    _alarm_dot(ax, dates, prr_lo, traj["evans"].to_numpy() & eligible, COLORS["prr"])
    with np.errstate(invalid="ignore"):
        _alarm_dot(ax, dates, eb05, (eb05 >= 2) & eligible, COLORS["gps"])
    ax.set_title("Cumulative disproportionality (ratio scale; lines at 1 and 2)")
    _legend(ax)

    ax = axes[2]
    ic, lo, hi = series("ic"), series("ic025"), series("ic975")
    ax.fill_between(dates, lo, hi, color=COLORS["bcpnn"], alpha=0.12, linewidth=0)
    ax.plot(dates, ic, color=COLORS["bcpnn"], label="BCPNN, IC with 95% interval")
    ax.axhline(0, color=REFERENCE, linewidth=1)
    with np.errstate(invalid="ignore"):
        _alarm_dot(ax, dates, ic, (lo > 0) & eligible, COLORS["bcpnn"])
    ax.set_title("Information component (signal when the interval clears 0)")
    _legend(ax)

    ax = axes[3]
    for name, label in (("hmm_prior", "HMM, prior parameters"), ("hmm_bw", "HMM, Baum-Welch")):
        belief = series(f"{name}_belief")
        ax.plot(dates, belief, color=COLORS[name], label=label)
        with np.errstate(invalid="ignore"):
            _alarm_dot(ax, dates, belief, (belief >= 0.5) & eligible, COLORS[name])
    ax.plot(dates, series("hmm_bw_smoothed"), color=COLORS["hmm_bw"], linewidth=1.4,
            linestyle=(0, (1, 2)), label="HMM, Baum-Welch, in hindsight\n(smoothed over the window)")
    ax.axhline(0.5, color=REFERENCE, linewidth=1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("HMM belief state, P(risky | reports so far)")
    _legend(ax)

    _mark_action(axes, action_date, action_label)
    fig.suptitle(title, x=0.125, ha="left", fontsize=12, fontweight="bold", y=0.955)
    fig.text(0.125, 0.045, "A dot marks the first quarter each method's conventional rule is met.",
             fontsize=8, color=INK_2)
    fig.savefig(path)
    plt.close(fig)


def lead_times(cases, method_list, path, target_far):
    """Lead time per case and method, conventional rules beside FAR-matched ones.

    A filled marker is an alarm that was still standing when the FDA acted (or
    that came after the action).  A hollow marker is an early alarm that had
    faded again by then.
    """
    drugs = list(dict.fromkeys(cases["drug"]))
    panels = [("conventional", "Conventional rules"),
              ("matched", f"Thresholds matched to {target_far:.0%} false alarms")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.95 * len(drugs) + 1.6), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    for ax, (kind, title) in zip(axes, panels):
        shown = [m for m in method_list if kind == "matched" or m.has_conventional]
        offsets = np.linspace(-0.36, 0.36, len(shown))
        for m, dy in zip(shown, offsets):
            rows = cases[(cases["method"] == m.label) & (cases["operating_point"] == kind)]
            rows = rows.set_index("drug").reindex(drugs)
            y = np.arange(len(drugs)) + dy
            lead = rows["lead_months"].to_numpy(dtype=float)
            faded = (lead > 0) & ~rows["standing_at_action"].fillna(False).to_numpy(dtype=bool)
            solid = np.isfinite(lead) & ~faded
            marker = "D" if m.key.endswith("_level") else "o"
            color = COLORS[m.family]
            ax.plot(lead[solid], y[solid], linestyle="none", marker=marker, ms=6.5, color=color,
                    mec=SURFACE, mew=1.5, label=m.label, zorder=3)
            ax.plot(lead[faded], y[faded], linestyle="none", marker=marker, ms=5.5,
                    mfc=SURFACE, mec=color, mew=1.8, zorder=3)
        ax.axvline(0, color=INK, linewidth=1, zorder=1)
        ax.set_title(title)
        ax.set_xlabel("Months before the FDA action (negative = after)")
        ax.set_yticks(np.arange(len(drugs)), [d.capitalize() for d in drugs])
        ax.grid(axis="y", visible=False)
        for edge in np.arange(len(drugs) + 1) - 0.5:
            ax.axhline(edge, color=GRID, linewidth=0.8)
    axes[0].invert_yaxis()
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.text(0.125, -0.03,
             "Hollow marker: an early alarm that had faded by the time the FDA acted.   "
             "No marker: no alarm inside the replay window.",
             fontsize=8, color=INK_2)
    fig.savefig(path)
    plt.close(fig)


def tradeoff(curve, method_list, path, n_cases):
    """Lead time and early detections against the false-alarm rate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.2})
    for m in method_list:
        block = curve[curve["method"] == m.label]
        if block.empty:
            continue
        # best achievable at or below each false-alarm rate
        block = block.sort_values("far")
        far = np.r_[0.0, block["far"].to_numpy()]
        style = {"color": COLORS[m.family], "label": m.label,
                 "linestyle": (0, (1, 2.2)) if m.key.endswith("_level") else "-",
                 "drawstyle": "steps-post"}
        for ax, col in zip(axes, ("mean_lead_months", "n_early")):
            best = np.maximum.accumulate(block[col].to_numpy())
            ax.plot(far, np.r_[0.0, best], **style)
    axes[0].set_title("Mean lead time over the FDA action")
    axes[0].set_ylabel("Months")
    axes[1].set_title(f"Cases flagged before the FDA action (of {n_cases})")
    axes[1].set_ylim(-0.2, n_cases + 0.2)
    for ax in axes:
        ax.set_xlabel("False-alarm rate on held-out negative controls")
        ax.set_xlim(0, curve["far"].max() * 1.02)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.savefig(path)
    plt.close(fig)


def convergence(history, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(np.arange(1, len(history) + 1), history, color=COLORS["hmm_bw"], marker="o", ms=4)
    ax.set_title("Baum-Welch training objective")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood plus log-prior")
    fig.savefig(path)
    plt.close(fig)


def masking(with_all, without, share, action_date, path):
    """Rosiglitazone -> MI with and without rofecoxib reports in the background."""
    dates = quarter_dates(with_all["quarter"])
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.2), sharex=True, gridspec_kw={"hspace": 0.34})

    axes[0].plot(dates, share, color=INK_2)
    axes[0].fill_between(dates, 0, share, color=INK_2, alpha=0.1, linewidth=0)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Share of all MI reports in the quarter that mention rofecoxib", pad=16)

    series = ((with_all, DATA_A, "All reports"), (without, DATA_B, "Rofecoxib reports removed"))
    for frame, color, label in series:
        ratio = (frame["a"] / frame["expected"]).to_numpy(dtype=float)
        axes[1].plot(dates, ratio, color=color, label=label)
        axes[2].plot(dates, frame["hmm_bw_belief"].to_numpy(dtype=float), color=color, label=label)
    axes[1].axhline(1, color=REFERENCE, linewidth=1)
    _plain_log(axes[1])
    axes[1].set_title("Rosiglitazone MI reports per quarter, observed / expected")
    _legend(axes[1])
    axes[2].axhline(0.5, color=REFERENCE, linewidth=1)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title("HMM (Baum-Welch) belief that rosiglitazone -> MI is elevated")
    _legend(axes[2])

    _mark_action(axes, action_date, "FDA alert")
    fig.savefig(path)
    plt.close(fig)


def stimulated_reporting(full, markers, path):
    """Full-history counts for one pair, showing reporting surges after publicity."""
    dates = quarter_dates(full["quarter"])
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.6), sharex=True, gridspec_kw={"hspace": 0.3})
    axes[0].bar(dates, full["a"], width=70, color=DATA_A, linewidth=0)
    axes[0].set_title("Rosiglitazone reports per quarter that mention MI", pad=30)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    ratio = (full["a"] / full["expected"]).where((full["n_drug"] >= 30) & (full["a"] > 0))
    axes[1].plot(dates, ratio.to_numpy(dtype=float), color=DATA_A, marker="o", ms=2.5)
    axes[1].axhline(1, color=REFERENCE, linewidth=1)
    _plain_log(axes[1])
    axes[1].set_title("Observed / expected MI reports in the quarter (shown when the drug has 30+ reports)")

    for i, (date, label) in enumerate(markers):
        date = pd.Timestamp(date)
        for ax in axes:
            ax.axvline(date, color=INK, linewidth=1, zorder=1)
        axes[0].annotate(label, xy=(date, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(3, 2 + 10 * (i % 2)), textcoords="offset points",
                         fontsize=8, va="bottom", ha="left")
    fig.savefig(path)
    plt.close(fig)
