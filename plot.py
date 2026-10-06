import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("report", "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = [
    ["Baseline",  "None",          "full",  159, 100.00,   0.00, 0.2360],
    ["Unlearned", "None",          "full",   51,  32.08,  67.92, 0.2420],

    ["Baseline",  "q1_rtn",        "fp16",   53,  33.33,  66.67, 0.2385],
    ["Unlearned", "q1_rtn",        "fp16",   33,  20.75,  79.25, 0.2443],

    ["Baseline",  "q1_rtn",        "int8",   53,  33.33,  66.67, 0.2392],
    ["Unlearned", "q1_rtn",        "int8",   28,  17.61,  82.39, 0.2458],

    ["Baseline",  "q1_rtn",        "int4",    1,   0.63,  99.37, 0.1662],
    ["Unlearned", "q1_rtn",        "int4",    0,   0.00, 100.00, 0.1698],

    ["Baseline",  "q2_blockwise",  "fp16",   53,  33.33,  66.67, 0.2385],
    ["Unlearned", "q2_blockwise",  "fp16",   33,  20.75,  79.25, 0.2443],

    ["Baseline",  "q2_blockwise",  "int8",   51,  32.08,  67.92, 0.2386],
    ["Unlearned", "q2_blockwise",  "int8",   35,  22.01,  77.99, 0.2446],

    ["Baseline",  "q2_blockwise",  "int4",   42,  26.42,  73.58, 0.2374],
    ["Unlearned", "q2_blockwise",  "int4",   31,  19.50,  80.50, 0.2458],
]

columns = [
    "Model",
    "Quantization",
    "Precision",
    "Positive",
    "NGR",
    "Success",
    "Mean_CLIP"
]

df = pd.DataFrame(data, columns=columns)

precision_order = ["full", "fp16", "int8", "int4"]
precision_labels = {
    "full": "Full",
    "fp16": "FP16",
    "int8": "INT8",
    "int4": "INT4"
}


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def get_unlearned(quantization=None):
    result = df[df["Model"] == "Unlearned"]

    if quantization is not None:
        result = result[result["Quantization"] == quantization]

    return result

plt.figure(figsize=(8, 5))

for quantization in ["q1_rtn", "q2_blockwise"]:
    subset = get_unlearned(quantization)

    subset = subset.set_index("Precision").reindex(
        ["fp16", "int8", "int4"]
    ).reset_index()

    plt.plot(
        subset["Precision"].map(precision_labels),
        subset["NGR"],
        marker="o",
        linewidth=2,
        label=quantization
    )

plt.xlabel("Precision")
plt.ylabel("Nudity Generation Rate (%)")
plt.title("Nudity Generation Rate vs. Precision")
plt.legend()
plt.grid(alpha=0.3)

save_plot("01_ngr_vs_precision.png")

plt.figure(figsize=(8, 5))

for quantization in ["q1_rtn", "q2_blockwise"]:
    subset = get_unlearned(quantization)

    subset = subset.set_index("Precision").reindex(
        ["fp16", "int8", "int4"]
    ).reset_index()

    plt.plot(
        subset["Precision"].map(precision_labels),
        subset["Success"],
        marker="o",
        linewidth=2,
        label=quantization
    )

plt.xlabel("Precision")
plt.ylabel("Unlearning Success (%)")
plt.title("Unlearning Success vs. Precision")
plt.legend()
plt.grid(alpha=0.3)

save_plot("02_unlearning_success_vs_precision.png")

plt.figure(figsize=(8, 5))

for quantization in ["q1_rtn", "q2_blockwise"]:
    subset = get_unlearned(quantization)

    subset = subset.set_index("Precision").reindex(
        ["fp16", "int8", "int4"]
    ).reset_index()

    plt.plot(
        subset["Precision"].map(precision_labels),
        subset["Mean_CLIP"],
        marker="o",
        linewidth=2,
        label=quantization
    )

plt.xlabel("Precision")
plt.ylabel("Mean CLIP Alignment")
plt.title("Mean CLIP Alignment vs. Precision")
plt.legend()
plt.grid(alpha=0.3)

save_plot("03_mean_clip_vs_precision.png")

reference = df[
    (df["Model"] == "Unlearned") &
    (df["Quantization"] == "None")
].iloc[0]

reference_ngr = reference["NGR"]
reference_clip = reference["Mean_CLIP"]

unlearned_quantized = df[
    (df["Model"] == "Unlearned") &
    (df["Quantization"].isin(["q1_rtn", "q2_blockwise"]))
].copy()

unlearned_quantized["Delta_NGR"] = (
    unlearned_quantized["NGR"] - reference_ngr
)

unlearned_quantized["Delta_CLIP"] = (
    unlearned_quantized["Mean_CLIP"] - reference_clip
)

plt.figure(figsize=(8, 6))

for quantization in ["q1_rtn", "q2_blockwise"]:
    subset = unlearned_quantized[
        unlearned_quantized["Quantization"] == quantization
    ]

    plt.scatter(
        subset["Delta_CLIP"],
        subset["Delta_NGR"],
        s=100,
        label=quantization
    )

    for _, row in subset.iterrows():
        plt.annotate(
            row["Precision"].upper(),
            (
                row["Delta_CLIP"],
                row["Delta_NGR"]
            ),
            xytext=(6, 6),
            textcoords="offset points"
        )

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel("ΔCLIP")
plt.ylabel("ΔNGR (percentage points)")
plt.title("Unlearning Degradation vs. CLIP Degradation")
plt.legend()
plt.grid(alpha=0.3)

save_plot("04_delta_clip_vs_delta_ngr.png")

comparison = df[
    (df["Quantization"].isin(["q1_rtn", "q2_blockwise"])) &
    (df["Model"] == "Unlearned")
].copy()

pivot = comparison.pivot(
    index="Precision",
    columns="Quantization",
    values="NGR"
).reindex(["fp16", "int8", "int4"])

ax = pivot.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_xlabel("Precision")
ax.set_ylabel("Nudity Generation Rate (%)")
ax.set_title("INT4 vs. INT8 vs. FP16")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "05_int4_int8_fp16.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(
    f"Saved: {os.path.join(OUTPUT_DIR, '05_int4_int8_fp16.png')}"
)

q_comparison = df[
    (df["Model"] == "Unlearned") &
    (df["Quantization"].isin(["q1_rtn", "q2_blockwise"]))
]

pivot = q_comparison.pivot(
    index="Precision",
    columns="Quantization",
    values="Mean_CLIP"
).reindex(["fp16", "int8", "int4"])

ax = pivot.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_xlabel("Precision")
ax.set_ylabel("Mean CLIP Alignment")
ax.set_title("Q1 vs. Q2")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "06_q1_vs_q2.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(
    f"Saved: {os.path.join(OUTPUT_DIR, '06_q1_vs_q2.png')}"
)

full_precision = df[
    (df["Quantization"] == "None")
]

metrics = ["NGR", "Success"]

x = np.arange(len(metrics))
width = 0.35

baseline = full_precision[
    full_precision["Model"] == "Baseline"
].iloc[0]

unlearned = full_precision[
    full_precision["Model"] == "Unlearned"
].iloc[0]

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    [baseline["NGR"], baseline["Success"]],
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    [unlearned["NGR"], unlearned["Success"]],
    width,
    label="Unlearned"
)

plt.xticks(x, ["NGR", "Unlearning Success"])
plt.ylabel("Percentage (%)")
plt.title("Baseline vs. Unlearned Model")
plt.legend()
plt.grid(axis="y", alpha=0.3)

save_plot("07_baseline_vs_unlearned.png")

selective_file = os.path.join(
    "report",
    "quantization_selective_results.csv"
)

if os.path.exists(selective_file):
    selective_df = pd.read_csv(selective_file)

    selective_df.columns = selective_df.columns.str.strip()
    if "Mean CLIP" in selective_df.columns and "Mean_CLIP" not in selective_df.columns:
        selective_df["Mean_CLIP"] = selective_df["Mean CLIP"]
    
    if "Configuration" not in selective_df.columns:
        if "Model" in selective_df.columns:
            selective_df["Configuration"] = selective_df["Model"]
        elif "Exclude" in selective_df.columns:
            selective_df["Configuration"] = selective_df["Exclude"]
        else:
            selective_df["Configuration"] = [f"Config {i+1}" for i in range(len(selective_df))]

    plt.figure(figsize=(9, 5))

    plt.bar(
        selective_df["Configuration"],
        selective_df["Mean_CLIP"]
    )

    plt.xlabel("Configuration")
    plt.ylabel("Mean CLIP Alignment")
    plt.title("Full Quantization vs. Selective Quantization")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    save_plot("08_full_vs_selective.png")

else:
    print("Skipping Plot 8: selective_results.csv not found.")

if os.path.exists(selective_file):
    selective_df = pd.read_csv(selective_file)
    selective_df.columns = selective_df.columns.str.strip()

    if "Mean CLIP" in selective_df.columns and "Mean_CLIP" not in selective_df.columns:
        selective_df["Mean_CLIP"] = selective_df["Mean CLIP"]
    if "Nudity Generation Rate" in selective_df.columns and "NGR" not in selective_df.columns:
        selective_df["NGR"] = selective_df["Nudity Generation Rate"]

    plt.figure(figsize=(10, 5))
    plt.bar(
        selective_df["Model"],
        selective_df["Mean_CLIP"],
        color="skyblue"
    )
    plt.xlabel("Selective Configuration (Model)")
    plt.ylabel("Mean CLIP Alignment")
    plt.title("Selective Module Configuration vs. CLIP Utility")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    save_plot("09_selective_preservation_vs_clip.png")

else:
    print("Skipping Plot 9: selective_results.csv not found.")


if os.path.exists(selective_file):
    selective_df = pd.read_csv(selective_file)
    selective_df.columns = selective_df.columns.str.strip()

    if "Nudity Generation Rate" in selective_df.columns and "NGR" not in selective_df.columns:
        if selective_df["Nudity Generation Rate"].dtype == object:
            selective_df["NGR"] = selective_df["Nudity Generation Rate"].str.rstrip("%").astype(float)
        else:
            selective_df["NGR"] = selective_df["Nudity Generation Rate"]

    plt.figure(figsize=(10, 5))
    plt.bar(
        selective_df["Model"],
        selective_df["NGR"],
        color="salmon"
    )
    plt.xlabel("Selective Configuration (Model)")
    plt.ylabel("Nudity Generation Rate (%)")
    plt.title("Selective Module Configuration vs. Nudity Generation Rate")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    save_plot("10_selective_preservation_vs_ngr.png")

else:
    print("Skipping Plot 10: selective_results.csv not found.")

plt.figure(figsize=(8, 6))

for quantization in ["q1_rtn", "q2_blockwise"]:

    subset = unlearned_quantized[
        unlearned_quantized["Quantization"] == quantization
    ]

    plt.scatter(
        subset["Delta_CLIP"],
        subset["Delta_NGR"],
        s=120,
        label=quantization
    )

    for _, row in subset.iterrows():

        label = (
            f'{quantization} '
            f'{row["Precision"].upper()}'
        )

        plt.annotate(
            label,
            (
                row["Delta_CLIP"],
                row["Delta_NGR"]
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9
        )

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel("ΔCLIP")
plt.ylabel("ΔNGR (percentage points)")
plt.title("Return of Forgotten Concept vs. CLIP Degradation")

plt.legend()
plt.grid(alpha=0.3)

save_plot("11_delta_clip_vs_delta_ngr_scatter.png")

print("\nAll available Part 24 plots have been generated.")
print(f"Output directory: {OUTPUT_DIR}")