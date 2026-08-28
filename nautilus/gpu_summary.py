"""Cluster GPU inventory: every model, its VRAM, and where it lives.

Names printed in the model column are exactly what `--gpu-type` and the
templates' `nvidia.com/gpu.product` affinity match on.
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List

# Widest `nvidia.com/gpu.product` on the cluster is 55 characters
# (NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition); truncating it
# would print a name no affinity can match.
MODEL_WIDTH = 56
NODE_WIDTH = 46
TABLE_WIDTH = MODEL_WIDTH + NODE_WIDTH + 24

# The sizes GPUs are actually sold in, for snapping a reported figure back
# to its nominal tier.
NOMINAL_VRAM_GB = (8, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 80, 94, 96,
                   141, 192)

#: VRAM by model substring, for nodes whose `nvidia.com/gpu.memory` label is
#: missing. Matched longest-first, so `A100-80GB-PCIe-MIG-1g.10gb` wins over
#: the `A100-80GB` it contains.
VRAM_BY_MODEL = {
    "A100-80GB": 80,
    "A100-SXM4-80GB": 80,
    "A100-80GB-PCIe": 80,
    "A100-80GB-PCIe-MIG-1g.10gb": 10,
    "A100-PCIE-40GB": 40,
    "RTX-A6000": 48,
    "RTX-A5000": 24,
    "RTX-A4000": 16,
    "GeForce-RTX-4090": 24,
    "GeForce-RTX-4080": 16,
    "GeForce-RTX-3090-Ti": 24,
    "GeForce-RTX-3090": 24,
    "GeForce-RTX-3080-Ti": 12,
    "GeForce-RTX-3080": 10,
    "GeForce-RTX-2080-Ti": 11,
    "GeForce-GTX-1080-Ti": 11,
    "GeForce-GTX-1080": 8,
    "A40": 48,
    "L40": 48,
    "L4": 24,
    "V100-SXM2-32GB": 32,
    "V100-PCIE-16GB": 16,
    "TITAN-RTX": 24,
    "TITAN-Xp": 12,
    "A10": 24,
    "T4": 16,
    "M10": 8,
    # The 480 in the name is system LPDDR; a job gets the 96 GB of HBM.
    "GH200-480GB": 96,
    "Quadro-RTX-8000": 48,
    "Quadro-RTX-6000": 24,
    "Quadro-M4000": 8,
    "H200-NVL": 141,
    "RTX-PRO-6000-Blackwell": 96,
    "RTX-5000-Ada-Generation": 32,
    "RTX-4000-Ada-Generation": 20,
    "TITAN-X-Pascal": 12,
    "A2": 16,
}


def run_command(cmd: List[str]) -> str:
    """Run a command and return its stdout, empty on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        return ""


def node_vram_gb(labels: Dict[str, str]) -> int:
    """VRAM in GB from the node's own `nvidia.com/gpu.memory` label.

    The device plugin advertises USABLE memory in MiB, which is always a
    little under nominal -- a 3090 reports 24576 but an A10 reports 23028,
    both 24 GB cards. Rounding would split one tier across two rows, so the
    figure is snapped up to the next entry of `NOMINAL_VRAM_GB`.

    Right for any card, including ones no name table has heard of, and more
    accurate than the names where the two disagree: `NVIDIA-GH200-480GB`
    reports 97871 MiB, because the 480 in its name is system LPDDR and not
    the 96 GB of HBM a job can use.

    Returns 0 when the label is absent or unparseable, the caller's cue to
    fall back to `extract_vram_from_model`.
    """
    raw = labels.get("nvidia.com/gpu.memory")
    if not raw:
        return 0
    try:
        gb = int(raw) / 1024
    except ValueError:
        return 0
    for tier in NOMINAL_VRAM_GB:
        if gb <= tier:
            return tier
    return int(round(gb))


def extract_vram_from_model(gpu_model: str) -> int:
    """VRAM in GB guessed from the model string, for nodes with no label."""
    for pattern in sorted(VRAM_BY_MODEL, key=len, reverse=True):
        if pattern in gpu_model:
            return VRAM_BY_MODEL[pattern]
    # e.g. "24GB", "80G" embedded in an unknown name.
    match = re.search(r"(\d+)G[B]?", gpu_model, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _node_gpus(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per GPU model on a node, or [] if it advertises none."""
    labels = node["metadata"].get("labels", {})
    try:
        gpu_count = int(labels.get("nvidia.com/gpu.count", 0))
    except ValueError:
        return []
    if gpu_count <= 0:
        return []

    models = [
        value
        for key, value in labels.items()
        if key == "nvidia.com/gpu.product"
        or key.startswith("nvidia.com/gpu.product.")
    ]
    row = {"node_name": node["metadata"]["name"], "gpu_count": gpu_count}
    if not models:
        return [{**row, "vram_gb": 0, "model": "Unknown GPU"}]
    return [
        {
            **row,
            "vram_gb": node_vram_gb(labels) or extract_vram_from_model(model),
            "model": model,
        }
        for model in models
    ]


def get_gpu_nodes_summary() -> List[Dict[str, Any]]:
    """Every GPU node the cluster advertises, one row per model."""
    print("Getting quick GPU summary...")

    cmd = ["kubectl", "get", "nodes", "-l", "nvidia.com/gpu.count",
           "-o", "json"]
    output = run_command(cmd)
    if not output:
        print("No nodes with GPU labels found.")
        return []

    try:
        nodes_data = json.loads(output)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

    return [
        row
        for node in nodes_data.get("items", [])
        for row in _node_gpus(node)
    ]


def print_gpu_table(gpu_details: List[Dict[str, Any]]) -> None:
    """Print the inventory grouped by VRAM, then summary statistics."""
    if not gpu_details:
        print("No GPU information found.")
        return

    print("\n" + "=" * TABLE_WIDTH)
    print("GPU DISCOVERY TABLE (Grouped by VRAM - Ascending)")
    print("=" * TABLE_WIDTH)

    vram_groups = defaultdict(list)
    for gpu in sorted(gpu_details, key=lambda x: x["vram_gb"]):
        vram_groups[gpu["vram_gb"]].append(gpu)

    for vram_gb in sorted(vram_groups):
        group = vram_groups[vram_gb]
        vram_str = f"{vram_gb}GB" if vram_gb > 0 else "Unknown"
        total = sum(gpu["gpu_count"] for gpu in group)
        models = len({gpu["model"] for gpu in group})
        print(
            f"\n{vram_str} VRAM ({len(group)} nodes, {total} GPUs, "
            f"{models} models):"
        )
        print("-" * TABLE_WIDTH)
        print(
            f"{'GPU VRAM':<12} {'GPU Model':<{MODEL_WIDTH}} "
            f"{'Node Name':<{NODE_WIDTH}} {'GPU Count':<10}"
        )
        print("-" * TABLE_WIDTH)
        for gpu in group:
            print(
                f"{vram_str:<12} {gpu['model']:<{MODEL_WIDTH}} "
                f"{gpu['node_name'][:NODE_WIDTH - 1]:<{NODE_WIDTH}} "
                f"{gpu['gpu_count']:<10}"
            )

    print("\n" + "=" * TABLE_WIDTH)
    print("SUMMARY STATISTICS:")
    print("=" * TABLE_WIDTH)

    vram_values = [g["vram_gb"] for g in gpu_details if g["vram_gb"] > 0]
    print(f"Total GPUs: {sum(g['gpu_count'] for g in gpu_details)}")
    print(f"Unique GPU models: {len({g['model'] for g in gpu_details})}")
    print(f"Unique nodes: {len({g['node_name'] for g in gpu_details})}")
    if vram_values:
        print(f"VRAM range: {min(vram_values)}GB - {max(vram_values)}GB")

    print("\nVRAM Distribution:")
    distribution = defaultdict(int)
    for gpu in gpu_details:
        distribution[gpu["vram_gb"]] += gpu["gpu_count"]
    for vram_gb, count in sorted(distribution.items()):
        vram_str = f"{vram_gb}GB" if vram_gb > 0 else "Unknown"
        print(f"  {vram_str}: {count} GPUs")


def main() -> None:
    """Print the cluster's GPU inventory."""
    print("Quick GPU Summary Script for Kubernetes Cluster")
    print("=" * 60)

    try:
        subprocess.run(
            ["kubectl", "version", "--client"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: kubectl is not available or not configured properly.")
        sys.exit(1)

    print_gpu_table(get_gpu_nodes_summary())


if __name__ == "__main__":
    main()
