"""
Quick GPU Summary Script for Kubernetes Cluster

This script provides a quick overview of GPU distribution in the cluster
without processing every node in detail.
"""

import subprocess
import json
import re
import sys
from typing import List, Dict
from collections import defaultdict


def run_command(cmd: List[str]) -> str:
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        return ""


def get_gpu_nodes_summary():
    """Get a quick summary of nodes with GPUs"""
    print("Getting quick GPU summary...")
    
    # Get all nodes with GPU labels
    cmd = ["kubectl", "get", "nodes", "-l", "nvidia.com/gpu.count", "-o", "json"]
    output = run_command(cmd)
    
    if not output:
        print("No nodes with GPU labels found.")
        return []
    
    try:
        nodes_data = json.loads(output)
        gpu_details = []
        
        for node in nodes_data.get('items', []):
            node_name = node['metadata']['name']
            labels = node['metadata'].get('labels', {})
            
            # Get GPU count
            gpu_count = 0
            for key, value in labels.items():
                if key == 'nvidia.com/gpu.count':
                    try:
                        gpu_count = int(value)
                        break
                    except ValueError:
                        pass
            
            if gpu_count > 0:
                # Get GPU models
                gpu_models = []
                for key, value in labels.items():
                    if key == 'nvidia.com/gpu.product':
                        gpu_models.append(value)
                    elif key.startswith('nvidia.com/gpu.product.'):
                        gpu_models.append(value)
                
                if gpu_models:
                    for model in gpu_models:
                        vram = extract_vram_from_model(model)
                        gpu_details.append({
                            'vram_gb': vram,
                            'model': model,
                            'node_name': node_name,
                            'gpu_count': gpu_count
                        })
                else:
                    # Unknown GPU model
                    gpu_details.append({
                        'vram_gb': 0,
                        'model': 'Unknown GPU',
                        'node_name': node_name,
                        'gpu_count': gpu_count
                    })
        
        return gpu_details
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []


def extract_vram_from_model(gpu_model: str) -> int:
    """Extract VRAM in GB from GPU model string"""
    # Common VRAM mappings
    vram_mappings = {
        'A100-80GB': 80,
        'A100-SXM4-80GB': 80,
        'A100-80GB-PCIe': 80,
        'A100-80GB-PCIe-MIG-1g.10gb': 10,
        'A100-PCIE-40GB': 40,
        'RTX-A6000': 48,
        'RTX-A5000': 24,
        'RTX-A4000': 16,
        'GeForce-RTX-4090': 24,
        'GeForce-RTX-4080': 16,
        'GeForce-RTX-3090-Ti': 24,
        'GeForce-RTX-3090': 24,
        'GeForce-RTX-3080-Ti': 12,
        'GeForce-RTX-3080': 10,
        'GeForce-RTX-2080-Ti': 11,
        'GeForce-GTX-1080-Ti': 11,
        'GeForce-GTX-1080': 8,
        'A40': 48,
        'L40': 48,
        'L4': 24,
        'V100-SXM2-32GB': 32,
        'V100-PCIE-16GB': 16,
        'TITAN-RTX': 24,
        'TITAN-Xp': 12,
        'A10': 24,
        'T4': 16,
        'M10': 8,
        'GH200-480GB': 480,
        'Quadro-RTX-8000': 48,
        'Quadro-RTX-6000': 24,
        'Quadro-M4000': 8,
    }
    
    # Try exact matches first
    for model_pattern, vram in vram_mappings.items():
        if model_pattern in gpu_model:
            return vram
    
    # Try regex patterns for common formats
    patterns = [
        r'(\d+)GB',  # e.g., "24GB", "80GB"
        r'(\d+)G',   # e.g., "24G", "80G"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, gpu_model, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 0


def print_gpu_table(gpu_details: List):
    """Print GPU information in a table format grouped by VRAM"""
    if not gpu_details:
        print("No GPU information found.")
        return
    
    # Sort by VRAM (ascending)
    sorted_gpus = sorted(gpu_details, key=lambda x: x['vram_gb'])
    
    print("\n" + "="*120)
    print("GPU DISCOVERY TABLE (Grouped by VRAM - Ascending)")
    print("="*120)
    
    # Group by VRAM
    vram_groups = defaultdict(list)
    for gpu in sorted_gpus:
        vram_groups[gpu['vram_gb']].append(gpu)
    
    # Print grouped table
    for vram_gb in sorted(vram_groups.keys()):
        gpus_in_group = vram_groups[vram_gb]
        
        # Print VRAM group header
        vram_str = f"{vram_gb}GB" if vram_gb > 0 else "Unknown"
        total_gpus_in_group = sum(gpu['gpu_count'] for gpu in gpus_in_group)
        unique_models_in_group = len(set(gpu['model'] for gpu in gpus_in_group))
        
        print(f"\n{vram_str} VRAM ({len(gpus_in_group)} nodes, {total_gpus_in_group} GPUs, {unique_models_in_group} models):")
        print("-" * 120)
        
        # Print table header for this group
        print(f"{'GPU VRAM':<12} {'GPU Model':<35} {'Node Name':<50} {'GPU Count':<10}")
        print("-" * 120)
        
        # Print table rows for this group
        for gpu in gpus_in_group:
            vram_str = f"{gpu['vram_gb']}GB" if gpu['vram_gb'] > 0 else "Unknown"
            model = gpu['model'][:34]  # Truncate if too long
            node_name = gpu['node_name'][:49]  # Truncate if too long
            
            print(f"{vram_str:<12} {model:<35} {node_name:<50} {gpu['gpu_count']:<10}")
    
    # Print summary statistics
    print("\n" + "="*120)
    print("SUMMARY STATISTICS:")
    print("="*120)
    
    total_gpus = sum(gpu['gpu_count'] for gpu in gpu_details)
    unique_models = len(set(gpu['model'] for gpu in gpu_details))
    unique_nodes = len(set(gpu['node_name'] for gpu in gpu_details))
    vram_values = [gpu['vram_gb'] for gpu in gpu_details if gpu['vram_gb'] > 0]
    
    print(f"Total GPUs: {total_gpus}")
    print(f"Unique GPU models: {unique_models}")
    print(f"Unique nodes: {unique_nodes}")
    
    if vram_values:
        vram_range = f"{min(vram_values)}GB - {max(vram_values)}GB"
        print(f"VRAM range: {vram_range}")
    
    # Print VRAM distribution
    print(f"\nVRAM Distribution:")
    vram_distribution = defaultdict(int)
    for gpu in gpu_details:
        vram_distribution[gpu['vram_gb']] += gpu['gpu_count']
    
    for vram_gb in sorted(vram_distribution.keys()):
        count = vram_distribution[vram_gb]
        vram_str = f"{vram_gb}GB" if vram_gb > 0 else "Unknown"
        print(f"  {vram_str}: {count} GPUs")


def main():
    """Main function"""
    print("Quick GPU Summary Script for Kubernetes Cluster")
    print("="*60)
    
    # Check if kubectl is available
    try:
        subprocess.run(["kubectl", "version", "--client"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: kubectl is not available or not configured properly.")
        sys.exit(1)
    
    # Get GPU details
    gpu_details = get_gpu_nodes_summary()
    
    # Print table
    print_gpu_table(gpu_details)


if __name__ == "__main__":
    main() 