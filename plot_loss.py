#!/usr/bin/env python3
"""Sync log files from runpod and plot training loss curves."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOCAL_LOGS = SCRIPT_DIR / "logs"
DEFAULT_SSH = "7eccm2gamj7h7y-64411faa@ssh.runpod.io"
DEFAULT_KEY = os.path.expanduser("~/.ssh/id_ed25519")
REMOTE_LOGS = "/workspace/parameter-golf/logs/"


def sync_logs(ssh_host: str, ssh_key: str) -> None:
    LOCAL_LOGS.mkdir(exist_ok=True)
    ssh_opts = ["-i", ssh_key, "-o", "StrictHostKeyChecking=no"]
    # scp with wildcard — use shell to expand the glob on remote side
    print(f"Syncing logs from {ssh_host}...")
    cmd = f'scp -O {" ".join(ssh_opts)} "{ssh_host}:{REMOTE_LOGS}*.txt" "{LOCAL_LOGS}/"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        # Check if it's just "no match" vs real error
        if "No such file" in result.stderr or "no match" in result.stderr.lower():
            print("No remote log files found.")
        else:
            print(f"scp output: {result.stderr.strip()}")
    else:
        print("Sync complete.\n")


def parse_log(filepath: Path) -> dict:
    """Extract step, train_loss, val_loss, val_bpb from a log file."""
    train_steps, train_losses = [], []
    val_steps, val_losses, val_bpbs = [], [], []

    with open(filepath) as f:
        for line in f:
            # Train loss lines: step:100/20000 train_loss:2.7491 ...
            m = re.match(r"step:(\d+)/\d+ train_loss:([\d.]+)", line)
            if m:
                train_steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue

            # Val loss lines: step:1000/20000 val_loss:2.2862 val_bpb:1.3540 ...
            m = re.match(r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                val_bpbs.append(float(m.group(3)))
                continue

            # Final result line
            m = re.match(r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", line)
            if m:
                val_steps.append(val_steps[-1] if val_steps else 0)
                val_losses.append(float(m.group(1)))
                val_bpbs.append(float(m.group(2)))

    return {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "val_bpbs": val_bpbs,
    }


def shorten_label(name: str) -> str:
    """Turn verbose auto-generated run names into readable short labels."""
    if name.startswith("ours_"):
        name = name[5:]
    # Extract key params and build a compact label
    parts = []
    replacements = [
        ("mlp_width", "w"), ("mlp_window", "w"), ("gate_rank", "g"),
        ("adapt_rank", "a"), ("private_mlp_rank", "p"), ("num_layers", "L"),
        ("attend_every", "att"), ("q_latent", "ql"), ("kv_latent", "kvl"),
        ("mtp_heads", "mtp"), ("model_dim", "d"), ("mlp_overlap", "ov"),
    ]
    for long, short in replacements:
        idx = name.find(long)
        if idx != -1:
            # Extract the value after the key name
            val_start = idx + len(long)
            val = ""
            for c in name[val_start:]:
                if c == '_':
                    break
                val += c
            if val and val != "0":
                parts.append(f"{short}{val}")
    return "_".join(parts) if parts else name[:50]


def plot(log_files: list[Path], metric: str = "bpb") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(16, 10),
                                            height_ratios=[1, 1])
    fig.suptitle("Parameter Golf Training Runs", fontsize=14, fontweight="bold")

    # Use a distinct color per run
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, filepath in enumerate(log_files):
        data = parse_log(filepath)
        if not data["train_steps"]:
            continue

        label = filepath.stem
        if label.startswith("ours_"):
            label = shorten_label(label)
        elif label == "baseline" or label == "baseline_proper":
            pass  # keep as-is
        color = colors[i % len(colors)]

        # Train loss — top chart
        ax_train.plot(data["train_steps"], data["train_losses"],
                      color=color, alpha=0.8, linewidth=1.5, label=label)

        # Val BPB — bottom chart
        if metric == "bpb" and data["val_bpbs"]:
            ax_val.plot(data["val_steps"], data["val_bpbs"],
                        "o-", color=color, markersize=6, linewidth=2, label=label)
            # Annotate final val_bpb
            if data["val_bpbs"]:
                ax_val.annotate(f"{data['val_bpbs'][-1]:.4f}",
                                (data["val_steps"][-1], data["val_bpbs"][-1]),
                                textcoords="offset points", xytext=(8, 0),
                                fontsize=8, color=color, fontweight="bold")
        elif data["val_losses"]:
            ax_val.plot(data["val_steps"], data["val_losses"],
                        "o-", color=color, markersize=6, linewidth=2, label=label)

    # Train chart formatting
    ax_train.set_ylabel("Train Loss", fontsize=11)
    ax_train.set_title("Training Loss", fontsize=11)
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(fontsize=8, loc="upper right", framealpha=0.9)
    # Cap y-axis to ignore early spikes
    if ax_train.get_lines():
        all_y = []
        for line in ax_train.get_lines():
            all_y.extend(line.get_ydata())
        if all_y:
            p95 = sorted(all_y)[int(len(all_y) * 0.95)]
            ax_train.set_ylim(top=min(p95 * 1.2, max(all_y)))

    # Val chart formatting
    val_label = "Val BPB" if metric == "bpb" else "Val Loss"
    ax_val.set_xlabel("Step", fontsize=11)
    ax_val.set_ylabel(val_label, fontsize=11)
    ax_val.set_title(f"Validation ({val_label})", fontsize=11)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(fontsize=8, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = SCRIPT_DIR / "loss_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Sync runpod logs and plot loss curves")
    parser.add_argument("--no-sync", action="store_true", help="Skip rsync, just plot local logs")
    parser.add_argument("--ssh", default=DEFAULT_SSH, help="SSH host for runpod")
    parser.add_argument("--key", default=DEFAULT_KEY, help="SSH key path")
    parser.add_argument("--metric", choices=["bpb", "loss"], default="bpb",
                        help="Plot val_bpb or val_loss (default: bpb)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only plot logs matching this substring")
    parser.add_argument("logs", nargs="*", help="Specific log files to plot (default: all in logs/)")
    args = parser.parse_args()

    if not args.no_sync:
        sync_logs(args.ssh, args.key)

    if args.logs:
        log_files = [Path(f) for f in args.logs]
    else:
        log_files = sorted(LOCAL_LOGS.glob("*.txt"))

    if args.filter:
        log_files = [f for f in log_files if args.filter in f.name]

    if not log_files:
        print("No log files found.")
        sys.exit(1)

    print(f"Plotting {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  {f.name}")
    print()

    plot(log_files, args.metric)


if __name__ == "__main__":
    main()
