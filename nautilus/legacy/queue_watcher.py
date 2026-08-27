#!/usr/bin/env python3
"""Re-attach to a Nautilus Job queue or print its status.

Use after launch_nautilus_pods.py was interrupted, or to inspect a batch.

Examples:

  python nautilus/training/queue_watcher.py --list-groups
  python nautilus/training/queue_watcher.py --label <group-id> -nl 200
  python nautilus/training/queue_watcher.py --prefix pls-vla-0304 --status
"""

# Must match launch_nautilus_pods.PLS_VLA_QUEUE_GROUP_LABEL_KEY
PLS_VLA_QUEUE_GROUP_LABEL_KEY = "pls_vla_queue_group"

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OurJobStatuses:
    active_names: List[str]
    completed_names: List[str]
    suspended_names: List[str]


def get_active_pod_count() -> int:
    """Count all non-terminal pods (Pending/Running) in the current namespace."""
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-o', 'json'],
            capture_output=True, text=True, check=True,
        )
        pods = json.loads(result.stdout)
        active = 0
        for pod in pods.get('items', []):
            phase = pod.get('status', {}).get('phase', '')
            if phase in ('Pending', 'Running', 'Unknown'):
                active += 1
        return active
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: failed to query namespace pods: {e}")
        return 0


def get_jobs_by_label(label_value: str) -> OurJobStatuses:
    """Query jobs by pls_vla_queue_group label."""
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'jobs',
             '-l', f'{PLS_VLA_QUEUE_GROUP_LABEL_KEY}={label_value}',
             '-o', 'json'],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: failed to query jobs: {e}")
        return OurJobStatuses([], [], [])

    return _classify_jobs(data)


def get_jobs_by_prefix(prefix: str) -> OurJobStatuses:
    """Query all jobs and filter by name prefix."""
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'jobs', '-o', 'json'],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: failed to query jobs: {e}")
        return OurJobStatuses([], [], [])

    data['items'] = [
        j for j in data.get('items', [])
        if j['metadata']['name'].startswith(prefix)
    ]
    return _classify_jobs(data)


def _classify_jobs(data: dict) -> OurJobStatuses:
    active, completed, suspended = [], [], []
    for job in data.get('items', []):
        name = job['metadata']['name']
        spec_suspended = job.get('spec', {}).get('suspend', False)

        conditions = job.get('status', {}).get('conditions', [])
        is_done = any(
            c.get('type') in ('Complete', 'Failed') and c.get('status') == 'True'
            for c in conditions
        )

        if is_done:
            completed.append(name)
        elif spec_suspended:
            suspended.append(name)
        else:
            active.append(name)

    return OurJobStatuses(active, completed, suspended)


def list_queue_groups() -> None:
    """List all distinct pls_vla_queue_group label values in the namespace."""
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'jobs',
             '-l', PLS_VLA_QUEUE_GROUP_LABEL_KEY,
             '-o', 'json'],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Failed to query jobs: {e}")
        return

    groups: dict = {}
    for job in data.get('items', []):
        gid = job['metadata'].get('labels', {}).get(PLS_VLA_QUEUE_GROUP_LABEL_KEY, '?')
        spec_suspended = job.get('spec', {}).get('suspend', False)
        conditions = job.get('status', {}).get('conditions', [])
        is_done = any(
            c.get('type') in ('Complete', 'Failed') and c.get('status') == 'True'
            for c in conditions
        )
        if gid not in groups:
            groups[gid] = {'active': 0, 'completed': 0, 'suspended': 0}
        if is_done:
            groups[gid]['completed'] += 1
        elif spec_suspended:
            groups[gid]['suspended'] += 1
        else:
            groups[gid]['active'] += 1

    if not groups:
        print("No pls_vla_queue_group jobs found in the namespace.")
        return

    print(f"{'Queue Group ID':<25} {'Active':>7} {'Done':>7} {'Suspended':>10} {'Total':>7}")
    print("-" * 60)
    for gid in sorted(groups):
        g = groups[gid]
        total = g['active'] + g['completed'] + g['suspended']
        print(f"{gid:<25} {g['active']:>7} {g['completed']:>7} {g['suspended']:>10} {total:>7}")


def compute_available_slots(namespace_pod_limit: int, max_concurrent: int,
                            active_pod_count: int, our_active_count: int) -> int:
    namespace_slots = namespace_pod_limit - active_pod_count
    if max_concurrent > 0:
        our_slots = max_concurrent - our_active_count
        return max(0, min(namespace_slots, our_slots))
    return max(0, namespace_slots)


def unsuspend_jobs(job_names: List[str]) -> int:
    ok = 0
    for name in job_names:
        try:
            subprocess.run(
                ['kubectl', 'patch', 'job', name, '-p', '{"spec":{"suspend":false}}'],
                capture_output=True, text=True, check=True,
            )
            print(f"  ▸ Unsuspended job: {name}")
            ok += 1
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or '').strip()
            print(f"  ✗ Failed to unsuspend {name}: {stderr or e}")
            if 'admission webhook' in stderr:
                remaining = len(job_names) - job_names.index(name) - 1
                if remaining > 0:
                    print(f"  ⚠ Skipping remaining {remaining} job(s) — webhook will deny them all for the same reason.")
                break
    return ok


def print_status(statuses: OurJobStatuses, active_pods: int, namespace_pod_limit: int) -> None:
    total = len(statuses.active_names) + len(statuses.completed_names) + len(statuses.suspended_names)
    print(f"Namespace pods: {active_pods}/{namespace_pod_limit}")
    print(f"Our jobs:  {len(statuses.active_names)} active, "
          f"{len(statuses.completed_names)} done, "
          f"{len(statuses.suspended_names)} suspended  "
          f"({total} total)")
    if statuses.active_names:
        print(f"  Active: {', '.join(sorted(statuses.active_names)[:10])}"
              + (f" ... (+{len(statuses.active_names)-10} more)" if len(statuses.active_names) > 10 else ""))
    if statuses.suspended_names:
        print(f"  Suspended: {', '.join(sorted(statuses.suspended_names)[:10])}"
              + (f" ... (+{len(statuses.suspended_names)-10} more)" if len(statuses.suspended_names) > 10 else ""))


def monitor_loop(get_statuses_fn, namespace_pod_limit: int, max_concurrent: int,
                 poll_interval: int = 30) -> None:
    """Poll loop that unsuspends jobs as namespace capacity frees up."""
    stop_event = threading.Event()

    def _handle_sigint(sig, frame):
        stop_event.set()

    prev_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while not stop_event.is_set():
            statuses = get_statuses_fn()
            active_pods = get_active_pod_count()
            total = len(statuses.active_names) + len(statuses.completed_names) + len(statuses.suspended_names)

            print(f"\n[Queue] Namespace pods: {active_pods}/{namespace_pod_limit} | "
                  f"Ours: {len(statuses.active_names)} running, "
                  f"{len(statuses.completed_names)} done, "
                  f"{len(statuses.suspended_names)} queued  "
                  f"({len(statuses.completed_names)}/{total} total complete)")

            if not statuses.suspended_names and not statuses.active_names:
                print("[Queue] All jobs finished.")
                break

            if stop_event.is_set():
                print(f"\n[Queue] Interrupted. {len(statuses.suspended_names)} jobs remain suspended.")
                break

            if statuses.suspended_names:
                slots = compute_available_slots(
                    namespace_pod_limit, max_concurrent,
                    active_pods, len(statuses.active_names),
                )
                if slots > 0:
                    to_unsuspend = statuses.suspended_names[:slots]
                    print(f"  Unsuspending {len(to_unsuspend)} job(s)...")
                    unsuspend_jobs(to_unsuspend)
                else:
                    print(f"  No slots available, waiting {poll_interval}s...")
            else:
                print(f"  All submitted, waiting for {len(statuses.active_names)} active job(s) to finish...")

            stop_event.wait(timeout=poll_interval)
    finally:
        signal.signal(signal.SIGINT, prev_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and unsuspend a Nautilus Job queue (or print status).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--label",
        help="Match jobs by pls_vla_queue_group label (from launch_nautilus_pods).",
    )
    group.add_argument(
        "--prefix",
        help="Match jobs by name prefix (e.g. pls-vla-0304).",
    )
    group.add_argument(
        "--list-groups",
        action="store_true",
        help="List queue groups in the namespace and exit.",
    )

    parser.add_argument(
        "-nl",
        "--namespace-pod-limit",
        type=int,
        default=200,
        help="Max active pods in the namespace (default: 200).",
    )
    parser.add_argument(
        "-mc",
        "--max-concurrent",
        type=int,
        default=0,
        help="Max of our jobs running at once (0 = namespace limit only).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status once and exit (do not monitor).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between polls while monitoring (default: 30).",
    )

    args = parser.parse_args()

    if args.list_groups:
        list_queue_groups()
        return

    if args.label:
        get_statuses_fn = lambda: get_jobs_by_label(args.label)
        selector_desc = f"label {PLS_VLA_QUEUE_GROUP_LABEL_KEY}={args.label}"
    else:
        get_statuses_fn = lambda: get_jobs_by_prefix(args.prefix)
        selector_desc = f"prefix '{args.prefix}'"

    if args.status:
        print(f"Querying jobs matching {selector_desc}...")
        statuses = get_statuses_fn()
        active_pods = get_active_pod_count()
        print_status(statuses, active_pods, args.namespace_pod_limit)
        return

    print(f"Monitoring jobs matching {selector_desc}")
    print(f"Namespace pod limit: {args.namespace_pod_limit}"
          + (f", max concurrent (ours): {args.max_concurrent}" if args.max_concurrent > 0 else ""))
    print(f"Poll interval: {args.poll_interval}s  |  Ctrl+C to detach")

    monitor_loop(get_statuses_fn, args.namespace_pod_limit, args.max_concurrent,
                 args.poll_interval)


if __name__ == '__main__':
    main()
