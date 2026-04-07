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
    ssh_opts = f"-i {ssh_key} -o StrictHostKeyChecking=no"
    # List remote log files first
    ls_cmd = ["ssh"] + ssh_opts.split() + [ssh_host, f"ls {REMOTE_LOGS}*.txt 2>/dev/null"]
    print(f"Syncing logs from {ssh_host}...")
    result = subprocess.run(ls_cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        print("No remote log files found.")
        return
    remote_files = result.stdout.strip().split("\n")
    # scp each file (runpod SSH doesn't support rsync)
    for rf in remote_files:
        fname = Path(rf).name
        local_path = LOCAL_LOGS / fname
        cmd = ["scp"] + ssh_opts.split() + [f"{ssh_host}:{rf}", str(local_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  {fname}")
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


def plot(log_files: list[Path], metric: str = "bpb") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(14, 7))

    for filepath in log_files:
        data = parse_log(filepath)
        if not data["train_steps"]:
            continue

        label = filepath.stem
        # Shorten long auto-generated names
        if label.startswith("ours_"):
            label = label[5:]
        # Truncate if still too long
        if len(label) > 60:
            label = label[:57] + "..."

        # Plot train loss
        ax.plot(data["train_steps"], data["train_losses"],
                alpha=0.6, linewidth=1, label=f"{label} (train)")

        # Plot val BPB or val loss as markers
        if metric == "bpb" and data["val_bpbs"]:
            ax.plot(data["val_steps"], data["val_bpbs"],
                    "o-", markersize=5, linewidth=2, label=f"{label} (val_bpb)")
        elif data["val_losses"]:
            ax.plot(data["val_steps"], data["val_losses"],
                    "o-", markersize=5, linewidth=2, label=f"{label} (val)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss / BPB")
    ax.set_title("Parameter Golf Training Runs")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = SCRIPT_DIR / "loss_plot.png"
    plt.savefig(out_path, dpi=150)
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
