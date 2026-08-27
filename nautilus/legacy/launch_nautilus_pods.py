#!/usr/bin/env python3
"""Launch PLS-VLA policy training on Nautilus (Kubernetes Pod or Job).

Fills a Pod/Job YAML template, injects a container training script (``lerobot-train``),
and submits via kubectl. Supports multi-seed repeats, namespace queueing (Jobs only),
PVC model persistence, and optional Hugging Face upload.

Examples (from repository root):

  python nautilus/training/launch_nautilus_pods.py --help
  python nautilus/training/launch_nautilus_pods.py -d <hf/dataset> -a act --dry_run
  python nautilus/training/launch_nautilus_pods.py -d <hf/dataset> -a act -j -nr 10 -nl 200 -mc 5

Resource names use the ``pls-vla-`` prefix; queue label is ``pls_vla_queue_group``.
Queueing (Jobs only): with -nl / -mc, overflow jobs submit suspended and are
unsuspended as slots free. Re-attach after Ctrl+C with queue_watcher.py --help.
"""

from __future__ import annotations

import copy
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import tyro
import yaml

LEROBOT_ALGOS = frozenset({"act", "pi05", "groot", "smolvla"})
# Kubernetes label for queue grouping (must match queue_watcher.py)
PLS_VLA_QUEUE_GROUP_LABEL_KEY = "pls_vla_queue_group"
# Pod/Job metadata.name prefix (DNS label-safe)
PLS_VLA_RESOURCE_PREFIX = "pls-vla"
# Wandb / lerobot --job_name prefix
PLS_VLA_JOB_NAME_PREFIX = "pls_vla"

# RFC 1123 DNS label max length; Pod hostname labels use this segment cap even though
# metadata.name may allow a longer DNS subdomain (253). We stay <=63 for portability.
K8S_DNS_LABEL_MAX_LEN = 63
_TRAINING_DIR = Path(__file__).resolve().parent


@dataclass
class NautilusPodConfig:
    """Nautilus training launcher.

    Pass --help for every flag. Prefer -j (Jobs) when using -nl / -mc queueing.
    """

    dataset: Annotated[
        Optional[str],
        tyro.conf.arg(
            name="dataset",
            help=(
                "Hugging Face dataset repo id (e.g. pollen-robotics/pick_and_place_bottle). "
                "Not required for --algo DUMMY."
            ),
            aliases=["-d"],
        ),
    ] = None

    algo: Annotated[
        str,
        tyro.conf.arg(
            name="algo",
            help="Policy to train: act | pi05 | groot | smolvla | DUMMY (sleep pod for cluster tests).",
            aliases=["-a"],
        ),
    ] = "act"

    repeat: Annotated[
        int,
        tyro.conf.arg(
            name="repeat",
            help="Number of independent runs with different random seeds.",
            aliases=["-nr"],
        ),
    ] = 1

    jobs: Annotated[
        bool,
        tyro.conf.arg(
            name="jobs",
            help="Submit as Kubernetes Jobs (required for -nl/-mc queueing). Default: Pods.",
            aliases=["-j"],
        ),
    ] = False

    yaml_file: Annotated[
        Optional[str],
        tyro.conf.arg(
            name="yaml_file",
            help="Path to Job or Pod YAML template (default: db-pls-vla-job/pod.yaml next to this script).",
            aliases=["-y"],
        ),
    ] = None

    namespace_pod_limit: Annotated[
        int,
        tyro.conf.arg(
            name="namespace_pod_limit",
            help=(
                "Max active pods in the namespace. Extra Jobs submit with suspend=true "
                "(0 = no limit). Jobs only (-j)."
            ),
            aliases=["-nl"],
        ),
    ] = 0

    max_concurrent: Annotated[
        int,
        tyro.conf.arg(
            name="max_concurrent",
            help="Cap on our own concurrent Jobs (0 = only namespace_pod_limit). Jobs only (-j).",
            aliases=["-mc"],
        ),
    ] = 0

    dry_run: Annotated[
        bool,
        tyro.conf.arg(
            name="dry_run",
            help="Print the generated container bash script and exit (no kubectl).",
        ),
    ] = False

    state_only_act: Annotated[
        bool,
        tyro.conf.arg(
            name="state_only_act",
            help=(
                "ACT on proprio-only datasets: remap observation.state → "
                "observation.environment_state."
            ),
        ),
    ] = False

    train_extra: Annotated[
        Optional[str],
        tyro.conf.arg(
            name="train_extra",
            help='Extra args appended to lerobot-train, as one quoted string (e.g. "--batch-size 8").',
        ),
    ] = None

    save_models: Annotated[
        bool,
        tyro.conf.arg(
            name="save_models",
            help="Copy lerobot-train outputs onto the PVC under --models_root.",
        ),
    ] = False

    models_root: Annotated[
        str,
        tyro.conf.arg(
            name="models_root",
            help="PVC directory for saved models when --save_models is set.",
        ),
    ] = "/nikola_vol/saved_models/pls-vla"

    upload_to_hub: Annotated[
        bool,
        tyro.conf.arg(
            name="upload_to_hub",
            help="After training, upload each model folder to --hf_model_repo.",
        ),
    ] = False

    hf_model_repo: Annotated[
        Optional[str],
        tyro.conf.arg(
            name="hf_model_repo",
            help="HF model repo id for uploads (e.g. erl-hub/reachy-act). Required with --upload_to_hub.",
        ),
    ] = None

    suffix: Annotated[
        str,
        tyro.conf.arg(
            name="suffix",
            help="Optional suffix on K8s pod/job names and training --job_name.",
            aliases=["-x"],
        ),
    ] = ""


@dataclass
class ContainerSpec:
    name: str
    shell_body: str

    def to_k8s_container(self, template_container: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(template_container)
        out["name"] = self.name
        out["command"] = ["/bin/bash", "-c"]
        out["args"] = [f"set -euo pipefail && \\\n{self.shell_body}"]
        return out


def _normalize_algo(raw: str) -> str:
    a = raw.strip().upper()
    if a == "DUMMY":
        return "DUMMY"
    lower = raw.strip().lower()
    if lower in LEROBOT_ALGOS:
        return lower
    raise ValueError(f"Unknown --algo {raw!r}; expected one of act, pi05, groot, smolvla, DUMMY")


def _dataset_slug(dataset: str) -> str:
    return dataset.replace("/", "-").replace("_", "-").lower()


def _resource_suffix_fragment(s: str) -> str:
    """RFC 1123-ish fragment for DNS labels; empty if s is blank after strip."""
    t = s.strip().lower()
    if not t:
        return ""
    t = "".join(c if (c.isalnum() or c in "-_") else "-" for c in t)
    while "--" in t:
        t = t.replace("--", "-")
    return t.strip("-")


def _slug_segment_fit(slug_n: str, max_slug_chars: int) -> str:
    """Return slug truncated to max_slug_chars (no leading hyphen)."""
    if max_slug_chars <= 0 or not slug_n:
        return ""
    if len(slug_n) <= max_slug_chars:
        return slug_n
    return slug_n[:max_slug_chars].rstrip("-")


def make_pls_vla_job_name(algo: str, seed: int, suffix: str, max_len: int = K8S_DNS_LABEL_MAX_LEN) -> str:
    """Wandb / training job id; <= max_len.

    Order: ``pls_vla_{algo}_{suffix}_s{seed}``; truncates suffix before seed if needed.
    """
    sfx = _resource_suffix_fragment(suffix).replace("-", "_")
    tail = f"s{seed}"
    if not sfx:
        name = f"{PLS_VLA_JOB_NAME_PREFIX}_{algo}_{tail}"
        return name[:max_len]
    name = f"{PLS_VLA_JOB_NAME_PREFIX}_{algo}_{sfx}_{tail}"
    if len(name) <= max_len:
        return name
    prefix = f"{PLS_VLA_JOB_NAME_PREFIX}_{algo}_"
    mid_tail = f"_{tail}"
    room = max_len - len(prefix) - len(mid_tail)
    if room <= 0:
        return f"{prefix}{tail}"[:max_len]
    sfx_t = sfx[:room].rstrip("_")
    return f"{prefix}{sfx_t}_{tail}" if sfx_t else f"{prefix}{tail}"[:max_len]


def _bash_single_quote(s: str) -> str:
    """Safe inside bash single-quoted string."""
    return s.replace("'", "'\"'\"'")


def build_training_script(
    dataset: str,
    policy_type: str,
    job_name: str,
    seed: int,
    state_only_act: bool = False,
    train_extra: Optional[str] = None,
    save_models: bool = False,
    models_root: str = "/nikola_vol/saved_models/pls-vla",
    upload_to_hub: bool = False,
    hf_model_repo: Optional[str] = None,
    suffix: str = "",
) -> str:
    """Bash body (after set -e): conda env, ffmpeg, pip extras, convert, train (Nautilus image)."""
    repo = _bash_single_quote(dataset)
    if policy_type == "act":
        pip_install = ":"
    elif policy_type == "groot":
        pip_install = ":"
    elif policy_type == "pi05":
        pip_install = ":"
    elif policy_type == "smolvla":
        pip_install = ":"
    else:
        raise ValueError(policy_type)
    # lerobot>=~0.5: convert lives under lerobot.scripts; 0.4.x uses lerobot.datasets.v30
    convert_dual = (
        f"if python -c \"import importlib.util as u, sys; "
        f"sys.exit(0 if u.find_spec('lerobot.scripts.convert_dataset_v21_to_v30') else 1)\"; then \\\n"
        f"  python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id='{repo}' --push-to-hub 0; \\\n"
        f"else \\\n"
        f"  python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id='{repo}' --push-to-hub 0; \\\n"
        f"fi"
    )
    dataset_slug = _dataset_slug(dataset)
    suffix_seg = _resource_suffix_fragment(suffix)
    train_dir_mid = (
        f"-{suffix_seg}" if suffix_seg else ""
    )
    main_cmd = (
        f"hf download {repo} --repo-type dataset --local-dir /home/user_lerobot/{dataset_slug} && "
        f"lerobot-train --dataset.repo_id='{repo}' --dataset.root=/home/user_lerobot/{dataset_slug} --policy.type={policy_type} --job_name={job_name} "
        f"--wandb.enable=true --policy.device=cuda --policy.push_to_hub=false"
    )
    if policy_type == "act" and state_only_act:
        main_cmd += " --rename_map='{\"observation.state\":\"observation.environment_state\"}'"
    if train_extra:
        main_cmd += f" {train_extra.strip()}"
    models_root_q = _bash_single_quote(models_root)
    if save_models or upload_to_hub:
        main_cmd = (
            f"mkdir -p '{models_root_q}' && "
            f"run_stamp=$(date +%Y-%m-%d_%H-%M-%S) && "
            f"train_output_dir='{models_root_q}'/${{run_stamp}}-{policy_type}{train_dir_mid}-{dataset_slug}_s{seed} && "
            f"{main_cmd} --output_dir=\"$train_output_dir\""
        )

    if upload_to_hub:
        if not hf_model_repo:
            raise ValueError("--hf_model_repo is required when --upload_to_hub is enabled")
        hf_repo_q = _bash_single_quote(hf_model_repo)
        main_cmd += (
            " && "
            "if [ -z \"${train_output_dir:-}\" ]; then "
            "echo 'train_output_dir was not set; cannot upload to HF' >&2; exit 1; "
            "fi && "
            "export train_output_dir && "
            "if [ -z \"${HF_TOKEN:-}\" ]; then "
            "echo 'Missing HF_TOKEN (set in pod env, e.g. from a Kubernetes secret)' >&2; exit 1; "
            "fi && "
            "uv pip install -q huggingface_hub && "
            f"HF_MODEL_REPO='{hf_repo_q}' "
            "python - <<'PY'\n"
            "import os\n"
            "from huggingface_hub import HfApi\n\n"
            "output_dir = os.environ['train_output_dir']\n"
            "repo_id = os.environ['HF_MODEL_REPO']\n"
            "token = os.environ['HF_TOKEN']\n"
            "path_in_repo = os.path.basename(output_dir.rstrip('/'))\n\n"
            "api = HfApi()\n"
            "api.upload_folder(\n"
            "    folder_path=output_dir,\n"
            "    path_in_repo=path_in_repo,\n"
            "    repo_id=repo_id,\n"
            "    token=token,\n"
            "    repo_type='model',\n"
            ")\n"
            "print(f'Uploaded {output_dir} -> {repo_id}/{path_in_repo}')\n"
            "PY"
        )

    return f"""source /lerobot/.venv/bin/activate && \\
{pip_install} && \\
{convert_dual} && \\
{main_cmd}"""


def build_dummy_script() -> str:
    return "sleep 86400"


class NautilusResource:
    def __init__(self, name: str, containers: List[ContainerSpec], is_job: bool = False):
        self.name = name
        self.containers = containers
        self.is_job = is_job

    def generate_yaml_config(
        self,
        template_file: str,
        suspend: bool = False,
        queue_group_label: Optional[str] = None,
    ) -> str:
        with open(template_file, "r") as f:
            config = yaml.safe_load(f)

        config["metadata"]["name"] = self.name
        if queue_group_label:
            config.setdefault("metadata", {}).setdefault("labels", {})
            config["metadata"]["labels"][PLS_VLA_QUEUE_GROUP_LABEL_KEY] = queue_group_label

        if self.is_job:
            spec_inner = config["spec"]["template"]["spec"]
            base_c = spec_inner["containers"][0]
            spec_inner["containers"] = [c.to_k8s_container(base_c) for c in self.containers]
            config["spec"]["backoffLimit"] = 25
            if suspend:
                config["spec"]["suspend"] = True
            else:
                config["spec"].pop("suspend", None)
        else:
            spec_inner = config["spec"]
            base_c = spec_inner["containers"][0]
            spec_inner["containers"] = [c.to_k8s_container(base_c) for c in self.containers]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(config, tmp, default_flow_style=False, sort_keys=False)
            return tmp.name

    def launch(
        self,
        template_file: str,
        suspend: bool = False,
        queue_group_label: Optional[str] = None,
    ) -> bool:
        path = self.generate_yaml_config(
            template_file, suspend=suspend, queue_group_label=queue_group_label
        )
        try:
            subprocess.run(["kubectl", "apply", "-f", path], check=True)
            kind = "job" if self.is_job else "pod"
            state = " (suspended)" if suspend else ""
            print(f"Launched {kind}{state}: {self.name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to launch {self.name}: {e}")
            return False
        finally:
            os.unlink(path)


class NautilusPod(NautilusResource):
    def __init__(self, name: str, containers: List[ContainerSpec]):
        super().__init__(name, containers, is_job=False)


class NautilusJob(NautilusResource):
    def __init__(self, name: str, containers: List[ContainerSpec]):
        super().__init__(name, containers, is_job=True)


def make_resource_name(
    ts: str,
    algo: str,
    slug: str,
    repeat_idx: int,
    repeat_total: int,
    seed: int,
    suffix: str = "",
    max_len: int = K8S_DNS_LABEL_MAX_LEN,
) -> str:
    """Pod/Job metadata.name <= max_len (63).

    Order: pls-vla, timestamp, algo, user suffix, literal s plus seed, repeat marker, then dataset slug at the end.
    Truncation preference: dataset slug, then suffix, then timestamp (shorter ts variants); algo and seed stay.
    """
    sfx = _resource_suffix_fragment(suffix)
    slug_n = slug or ""
    algo_p = algo.lower().replace("_", "-")
    repeat_s = f"-r{repeat_idx + 1}of{repeat_total}" if repeat_total > 1 else ""
    tail = f"s{seed}{repeat_s}"
    ts_candidates = [ts, ts.replace("-", ""), ""]

    def stem_no_slug(left: str, sfx_part: str) -> str:
        if sfx_part:
            return f"{left}{sfx_part}-{tail}"
        return f"{left}{tail}"

    def pack_one_ts(ts_use: str) -> Optional[tuple[str, str, str]]:
        """Return (stem_without_slug_suffix, sfx_use, slug_use) or None if unusable for this ts."""
        left = (
            f"{PLS_VLA_RESOURCE_PREFIX}-{ts_use}-{algo_p}-"
            if ts_use
            else f"{PLS_VLA_RESOURCE_PREFIX}-{algo_p}-"
        )
        if not sfx:
            stem = stem_no_slug(left, "")
            if len(stem) > max_len:
                return None
            room_slug = max_len - len(stem) - 1
            slug_use = _slug_segment_fit(slug_n, room_slug)
            return stem, "", slug_use

        for sfx_len in range(len(sfx), 0, -1):
            sfx_part = sfx[:sfx_len].rstrip("-")
            if not sfx_part:
                continue
            stem = stem_no_slug(left, sfx_part)
            if len(stem) > max_len:
                continue
            room_slug = max_len - len(stem) - 1
            slug_use = _slug_segment_fit(slug_n, room_slug)
            return stem, sfx_part, slug_use
        return None

    packed: Optional[tuple[str, str, str]] = None
    for ts_use in ts_candidates:
        packed = pack_one_ts(ts_use)
        if packed is not None:
            break

    if packed is None:
        left = f"pv-{algo_p}-"
        stem_em: str
        sfx_use: str = ""
        slug_use: str = ""
        if sfx:
            for sfx_len in range(len(sfx), 0, -1):
                sfx_part = sfx[:sfx_len].rstrip("-")
                if not sfx_part:
                    continue
                stem_try = stem_no_slug(left, sfx_part)
                if len(stem_try) <= max_len:
                    stem_em = stem_try
                    sfx_use = sfx_part
                    room_slug = max_len - len(stem_em) - 1
                    slug_use = _slug_segment_fit(slug_n, room_slug)
                    break
            else:
                stem_em = f"{left}{tail}"[:max_len].rstrip("-")
                room_slug = max_len - len(stem_em) - 1
                slug_use = _slug_segment_fit(slug_n, room_slug)
        else:
            stem_em = stem_no_slug(left, "")
            stem_em = stem_em[:max_len].rstrip("-")
            room_slug = max_len - len(stem_em) - 1
            slug_use = _slug_segment_fit(slug_n, room_slug)

        name = f"{stem_em}-{slug_use}" if slug_use else stem_em
        if len(name) > max_len:
            name = name[:max_len].rstrip("-")
        return name.strip("-")

    stem, _, slug_use = packed
    name = f"{stem}-{slug_use}" if slug_use else stem
    if len(name) > max_len:
        name = name[:max_len].rstrip("-")
    return name.strip("-")


# --- Queue (label PLS_VLA_QUEUE_GROUP_LABEL_KEY; must match queue_watcher.py) ---


def get_active_pod_count() -> int:
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-o", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        pods = json.loads(result.stdout)
        n = 0
        for pod in pods.get("items", []):
            phase = pod.get("status", {}).get("phase", "")
            if phase in ("Pending", "Running", "Unknown"):
                n += 1
        return n
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: failed to query pods: {e}")
        return 0


@dataclass
class OurJobStatuses:
    active_names: List[str]
    completed_names: List[str]
    suspended_names: List[str]


def get_our_job_statuses(queue_group_id: str) -> OurJobStatuses:
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "jobs",
                "-l",
                f"{PLS_VLA_QUEUE_GROUP_LABEL_KEY}={queue_group_id}",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: failed to query jobs: {e}")
        return OurJobStatuses([], [], [])

    active, completed, suspended = [], [], []
    for job in data.get("items", []):
        name = job["metadata"]["name"]
        spec_suspended = job.get("spec", {}).get("suspend", False)
        conditions = job.get("status", {}).get("conditions", [])
        is_done = any(
            c.get("type") in ("Complete", "Failed") and c.get("status") == "True" for c in conditions
        )
        if is_done:
            completed.append(name)
        elif spec_suspended:
            suspended.append(name)
        else:
            active.append(name)
    return OurJobStatuses(active, completed, suspended)


def compute_available_slots(
    namespace_pod_limit: int,
    max_concurrent: int,
    active_pod_count: int,
    our_active_count: int,
) -> int:
    ns_slots = namespace_pod_limit - active_pod_count
    if max_concurrent > 0:
        our_slots = max_concurrent - our_active_count
        return max(0, min(ns_slots, our_slots))
    return max(0, ns_slots)


def unsuspend_jobs(job_names: List[str]) -> int:
    ok = 0
    for name in job_names:
        try:
            subprocess.run(
                ["kubectl", "patch", "job", name, "-p", '{"spec":{"suspend":false}}'],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"  Unsuspended job: {name}")
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  Failed to unsuspend {name}: {e}")
    return ok


def monitor_and_unsuspend(
    queue_group_id: str,
    namespace_pod_limit: int,
    max_concurrent: int,
    poll_interval: int = 30,
) -> None:
    stop_event = threading.Event()

    def _sigint(_s, _f):
        stop_event.set()

    prev = signal.signal(signal.SIGINT, _sigint)
    try:
        while not stop_event.is_set():
            statuses = get_our_job_statuses(queue_group_id)
            active_pods = get_active_pod_count()
            total_ours = (
                len(statuses.active_names)
                + len(statuses.completed_names)
                + len(statuses.suspended_names)
            )
            print(
                f"\n[Queue] Namespace pods: {active_pods}/{namespace_pod_limit} | "
                f"Ours: {len(statuses.active_names)} running, "
                f"{len(statuses.completed_names)} done, "
                f"{len(statuses.suspended_names)} queued "
                f"({len(statuses.completed_names)}/{total_ours} complete)"
            )
            if not statuses.suspended_names and not statuses.active_names:
                print("[Queue] All jobs finished.")
                break
            if stop_event.is_set():
                _qw = _TRAINING_DIR / "queue_watcher.py"
                print(
                    f"\n[Queue] Interrupted. {len(statuses.suspended_names)} jobs suspended. "
                    f"Re-attach: python {_qw} "
                    f"--label {queue_group_id} -nl {namespace_pod_limit}"
                )
                break
            if statuses.suspended_names:
                slots = compute_available_slots(
                    namespace_pod_limit,
                    max_concurrent,
                    active_pods,
                    len(statuses.active_names),
                )
                if slots > 0:
                    batch = statuses.suspended_names[:slots]
                    print(f"  Unsuspending {len(batch)} job(s)...")
                    unsuspend_jobs(batch)
                else:
                    print(f"  No slots, waiting {poll_interval}s...")
            else:
                print(f"  Waiting for {len(statuses.active_names)} active job(s)...")
            stop_event.wait(timeout=poll_interval)
    finally:
        signal.signal(signal.SIGINT, prev)


def main() -> None:
    cfg = tyro.cli(NautilusPodConfig)
    algo = _normalize_algo(cfg.algo)

    if algo != "DUMMY" and not cfg.dataset:
        print("Error: --dataset is required unless --algo DUMMY", file=sys.stderr)
        sys.exit(1)
    if cfg.upload_to_hub and not cfg.hf_model_repo:
        print("Error: --hf_model_repo is required with --upload_to_hub", file=sys.stderr)
        sys.exit(1)

    use_job = cfg.jobs and algo != "DUMMY"
    nautilus_type = "job" if use_job else "pod"
    if cfg.yaml_file is None:
        yaml_path = str(_TRAINING_DIR / f"db-pls-vla-{nautilus_type}.yaml")
    else:
        yaml_path = cfg.yaml_file

    ts = datetime.now().strftime("%m%d-%H%M")
    queue_group_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    queuing = use_job and cfg.namespace_pod_limit > 0
    slug = _dataset_slug(cfg.dataset) if cfg.dataset else ""

    if cfg.dry_run:
        for i in range(cfg.repeat):
            seed = random.randint(1, 10000)
            if algo == "DUMMY":
                body = build_dummy_script()
                print(f"=== DUMMY run {i + 1}/{cfg.repeat} seed={seed} ===\n{body}\n")
            else:
                job_name = make_pls_vla_job_name(algo, seed, cfg.suffix)
                body = build_training_script(
                    cfg.dataset,
                    algo,
                    job_name,
                    seed,
                    state_only_act=cfg.state_only_act,
                    train_extra=cfg.train_extra,
                    save_models=cfg.save_models,
                    models_root=cfg.models_root,
                    upload_to_hub=cfg.upload_to_hub,
                    hf_model_repo=cfg.hf_model_repo,
                    suffix=cfg.suffix,
                )
                print(f"=== {algo} run {i + 1}/{cfg.repeat} job_name={job_name} ===\n{body}\n")
        return

    actual = cfg.repeat
    print(f"Launching {actual} {nautilus_type}(s) | template {yaml_path}")
    if algo == "DUMMY":
        print("  DUMMY: sleep pod (jobs flag ignored)")
    else:
        print(f"  dataset={cfg.dataset} algo={algo}")
        if cfg.upload_to_hub:
            print(f"  upload_to_hub=true hf_model_repo={cfg.hf_model_repo} (expects HF_TOKEN in pod env)")

    initial_slots = actual
    if queuing:
        ap = get_active_pod_count()
        initial_slots = compute_available_slots(
            cfg.namespace_pod_limit, cfg.max_concurrent, ap, 0
        )
        print(
            f"[Queue] Pods {ap}/{cfg.namespace_pod_limit} | slots {initial_slots} | group {queue_group_id}"
        )

    launched_active = 0
    launched_suspended = 0

    for i in range(actual):
        seed = random.randint(1, 10000)
        name = make_resource_name(ts, algo, slug, i, cfg.repeat, seed, cfg.suffix)

        if algo == "DUMMY":
            shell = build_dummy_script()
        else:
            job_name = make_pls_vla_job_name(algo, seed, cfg.suffix)
            shell = build_training_script(
                cfg.dataset,
                algo,
                job_name,
                seed,
                state_only_act=cfg.state_only_act,
                train_extra=cfg.train_extra,
                save_models=cfg.save_models,
                models_root=cfg.models_root,
                upload_to_hub=cfg.upload_to_hub,
                hf_model_repo=cfg.hf_model_repo,
                suffix=cfg.suffix,
            )

        ctr = ContainerSpec(name=f"pls-vla-{algo.lower()}-s{seed}", shell_body=shell)

        if use_job:
            res: NautilusResource = NautilusJob(name, [ctr])
        else:
            res = NautilusPod(name, [ctr])

        suspend = queuing and (launched_active >= initial_slots)
        ok = res.launch(
            yaml_path,
            suspend=suspend,
            queue_group_label=queue_group_id if queuing else None,
        )
        if ok:
            if suspend:
                launched_suspended += 1
            else:
                launched_active += 1

    if queuing and launched_suspended > 0:
        print(f"\n[Queue] {launched_active} active + {launched_suspended} suspended; monitoring...")
        monitor_and_unsuspend(queue_group_id, cfg.namespace_pod_limit, cfg.max_concurrent)
    elif queuing:
        print(f"\n[Queue] All {launched_active} jobs started without queuing.")


if __name__ == "__main__":
    main()
