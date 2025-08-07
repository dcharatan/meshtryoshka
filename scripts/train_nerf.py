import subprocess
from pathlib import Path

EXPERIMENT = "[nerf_synthetic]"

DATASETS = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]

WORKSPACE_ROOT = Path("workspace_nerf")
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


def run_scene(scene: str) -> None:
    """
    Run a single scene sequentially by invoking the training CLI.
    The WORKSPACE env var is set inline, so no external export is required.
    """
    workspace = WORKSPACE_ROOT / scene
    workspace.mkdir(parents=True, exist_ok=True)

    hydra_args = f"+experiment={EXPERIMENT} dataset.scene={scene}"

    cmd = f"WORKSPACE={workspace} python3 -m meshtryoshka {hydra_args}"

    print(f"\n[RUN] {cmd}\n")
    completed = subprocess.run(cmd, shell=True)
    if completed.returncode != 0:
        raise SystemExit(f"Job failed for scene={scene} (exit={completed.returncode})")


def main() -> None:
    for scene in DATASETS:
        run_scene(scene)
    print("\n[OK] All scenes completed successfully.")


if __name__ == "__main__":
    main()
