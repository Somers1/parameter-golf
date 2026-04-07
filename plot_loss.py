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

# Add substrings here to exclude runs from the plot
EXCLUDE = [
    "w6144",
    "w3072_g16",
    "w3072_g32_p16",
    "w3072_g32_p32",
    "w3072_g64",
]


def sync_logs(ssh_host: str, ssh_key: str) -> None:
    LOCAL_LOGS.mkdir(exist_ok=True)
    ssh_opts = ["-i", ssh_key, "-o", "StrictHostKeyChecking=no"]
    print(f"Syncing logs from {ssh_host}...")
    cmd = f'scp -O {" ".join(ssh_opts)} "{ssh_host}:{REMOTE_LOGS}*.txt" "{LOCAL_LOGS}/"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
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
            m = re.match(r"step:(\d+)/\d+ train_loss:([\d.]+)", line)
            if m:
                train_steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue

            m = re.match(r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                val_bpbs.append(float(m.group(3)))
                continue

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
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed. Install with: pip install plotly")
        sys.exit(1)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Training Loss", "Validation (Val BPB)"),
        vertical_spacing=0.08,
    )

    for filepath in log_files:
        data = parse_log(filepath)
        if not data["train_steps"]:
            continue

        label = filepath.stem
        if label.startswith("ours_"):
            label = shorten_label(label)

        # Train loss — top chart
        fig.add_trace(
            go.Scatter(
                x=data["train_steps"], y=data["train_losses"],
                mode="lines", name=label,
                legendgroup=label, showlegend=True,
                line=dict(width=1.5),
                opacity=0.8,
            ),
            row=1, col=1,
        )

        # Val BPB — bottom chart
        vals = data["val_bpbs"] if metric == "bpb" else data["val_losses"]
        if vals:
            final_bpb = f" ({vals[-1]:.4f})"
            fig.add_trace(
                go.Scatter(
                    x=data["val_steps"], y=vals,
                    mode="lines+markers", name=label + final_bpb,
                    legendgroup=label, showlegend=True,
                    line=dict(width=2),
                    marker=dict(size=6),
                ),
                row=2, col=1,
            )

    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="Train Loss", row=1, col=1)
    fig.update_yaxes(title_text="Val BPB" if metric == "bpb" else "Val Loss", row=2, col=1)

    fig.update_layout(
        title="Parameter Golf Training Runs",
        height=900,
        width=1400,
        template="plotly_white",
        legend=dict(
            font=dict(size=11),
            groupclick="toggleitem",
        ),
        hovermode="x unified",
    )

    out_path = SCRIPT_DIR / "loss_plot.html"
    fig.write_html(str(out_path))
    print(f"Saved interactive plot to {out_path}")

    # Also open in browser
    import webbrowser
    webbrowser.open(f"file://{out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Sync runpod logs and plot loss curves")
    parser.add_argument("--no-sync", action="store_true", help="Skip rsync, just plot local logs")
    parser.add_argument("--ssh", default=DEFAULT_SSH, help="SSH host for runpod")
    parser.add_argument("--key", default=DEFAULT_KEY, help="SSH key path")
    parser.add_argument("--metric", choices=["bpb", "loss"], default="bpb",
                        help="Plot val_bpb or val_loss (default: bpb)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only plot logs matching this substring")
    parser.add_argument("--exclude", type=str, nargs="+", default=[],
                        help="Exclude logs matching any of these substrings")
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
    all_excludes = EXCLUDE + (args.exclude or [])
    if all_excludes:
        log_files = [f for f in log_files
                     if not any(ex in f.name for ex in all_excludes)]

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
