"""Generate zombie detection dataset by playing KAZ episodes and capturing
frames paired with ground-truth zombie bounding boxes.

Each distortion level gets its own set of episodes because zombie-specific
distortions (levels 3-4: color change, pixel noise) are applied at sprite
creation time, not at render time. Screen-wide distortions (stars, clouds,
heat haze) are applied per-frame by the VisualWrapper transform.

Usage:
    python -m zombie_detection.generate_dataset [--config zombie_detection/config.yaml]
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
import pygame
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import visual_utils as vu  # noqa: E402
from visual_utils import set_distortion_level, VisualWrapper  # noqa: E402


ZOMBIE_SPECIFIC_LEVELS = {3, 4}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_zombie_positions(env, bbox_w: int, bbox_h: int) -> np.ndarray:
    """Walk through wrapper chain to reach the raw KAZ env and read zombie rects."""
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    boxes = []
    for zombie in inner.zombie_list:
        boxes.append([zombie.rect.x, zombie.rect.y, bbox_w, bbox_h])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def create_env_for_generation(max_zombies: int, distortion_level: int):
    """Create a KAZ environment configured for dataset generation."""
    from pettingzoo.butterfly import knights_archers_zombies_v10
    import supersuit as ss

    set_distortion_level(level=distortion_level)

    env = knights_archers_zombies_v10.env(
        max_cycles=2500,
        num_archers=2,
        num_knights=0,
        max_zombies=max_zombies,
        vector_state=False,
        render_mode="rgb_array",
    )
    env = VisualWrapper(env)
    env.render_mode = None
    env = ss.black_death_v3(env)
    return env


def capture_frame(env) -> np.ndarray:
    """Capture the full-screen observation as (H, W, 3) uint8."""
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    screen = pygame.surfarray.pixels3d(inner.screen)
    frame = np.array(screen)
    return np.swapaxes(frame, 0, 1)  # (W,H,3) -> (H,W,3)


def relevant_distortion_levels(num_zombies: int, all_levels: list) -> list:
    """Return only the distortion levels that produce unique images for this frame."""
    if num_zombies > 0:
        return list(all_levels)
    return [l for l in all_levels if l not in ZOMBIE_SPECIFIC_LEVELS]


def generate_split(
    split_name: str,
    num_examples: int,
    output_dir: Path,
    cfg: dict,
):
    """Generate one dataset split (train / val / test).

    For each distortion level, runs separate episodes so that zombie-specific
    distortions (color change, pixel noise) are correctly applied at sprite
    creation time. Empty frames skip zombie-specific levels since they produce
    identical results to level 2.
    """
    bbox_w = cfg["zombie"]["bbox_width"]
    bbox_h = cfg["zombie"]["bbox_height"]
    max_zombies = cfg["dataset"]["max_zombies"]
    distortion_levels = cfg["dataset"]["distortion_levels"]
    max_empty = cfg["dataset"]["max_empty_frames"]
    min_per_count = cfg["dataset"]["min_examples_per_zombie_count"]
    sample_interval = cfg["dataset"]["sample_interval"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-level tracking
    count_tracker = defaultdict(int)  # zombie_count -> num base frames
    empty_count = 0
    total_base = 0
    global_idx = 0

    # Target per distortion level: divide total evenly
    examples_per_level = num_examples // len(distortion_levels)
    remainder = num_examples % len(distortion_levels)

    print(f"\n[{split_name}] Generating {num_examples} base frames -> {output_dir}")
    print(f"  {examples_per_level} per distortion level, {len(distortion_levels)} levels")

    for level_idx, level in enumerate(distortion_levels):
        level_target = examples_per_level + (1 if level_idx < remainder else 0)
        level_count = 0
        step_counter = 0

        level_bar = tqdm(
            total=level_target,
            desc=f"  [{split_name}] dist={level}",
            unit="frame",
        )

        env = create_env_for_generation(max_zombies, distortion_level=level)

        while level_count < level_target:
            seed = np.random.randint(0, 2**31)
            env.reset(seed=seed)

            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    break

                action = 5 if np.random.random() < 0.7 else env.action_space(agent).sample()
                env.step(action)

                step_counter += 1
                if step_counter % sample_interval != 0:
                    continue

                gt_boxes = get_zombie_positions(env, bbox_w, bbox_h)
                n_zombies = len(gt_boxes)

                if n_zombies == 0 and level in ZOMBIE_SPECIFIC_LEVELS:
                    continue

                if n_zombies == 0:
                    if empty_count >= max_empty:
                        continue
                    empty_count += 1

                frame = capture_frame(env)

                fname_obs = f"{global_idx:06d}_d{level}_obs.npy"
                fname_boxes = f"{global_idx:06d}_d{level}_zombies.npy"
                np.save(str(output_dir / fname_obs), frame)
                np.save(str(output_dir / fname_boxes), gt_boxes)

                count_tracker[n_zombies] += 1
                total_base += 1
                global_idx += 1
                level_count += 1
                level_bar.update(1)

                counts_str = " ".join(f"{k}z:{v}" for k, v in sorted(count_tracker.items()))
                level_bar.set_postfix_str(counts_str)

                if level_count >= level_target:
                    break

            if level_count >= level_target:
                break

        level_bar.close()
        env.close()

    # Check minimum per zombie count and generate more if needed
    for zcount in range(max_zombies + 1):
        deficit = min_per_count - count_tracker.get(zcount, 0)
        if deficit <= 0:
            continue
        if zcount == 0 and empty_count >= max_empty:
            continue

        deficit_bar = tqdm(
            total=deficit,
            desc=f"  [{split_name}] fill {zcount}-zombie deficit",
            unit="frame",
        )

        env = create_env_for_generation(max_zombies, distortion_level=0)
        filled = 0
        step_counter = 0

        while filled < deficit:
            seed = np.random.randint(0, 2**31)
            env.reset(seed=seed)

            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    break

                action = 5 if np.random.random() < 0.7 else env.action_space(agent).sample()
                env.step(action)

                step_counter += 1
                if step_counter % sample_interval != 0:
                    continue

                gt_boxes = get_zombie_positions(env, bbox_w, bbox_h)
                n_zombies = len(gt_boxes)

                if n_zombies != zcount:
                    continue

                if n_zombies == 0 and empty_count >= max_empty:
                    break

                frame = capture_frame(env)

                fname_obs = f"{global_idx:06d}_d0_obs.npy"
                fname_boxes = f"{global_idx:06d}_d0_zombies.npy"
                np.save(str(output_dir / fname_obs), frame)
                np.save(str(output_dir / fname_boxes), gt_boxes)

                count_tracker[n_zombies] += 1
                if n_zombies == 0:
                    empty_count += 1
                total_base += 1
                global_idx += 1
                filled += 1
                deficit_bar.update(1)

                if filled >= deficit:
                    break

        deficit_bar.close()
        env.close()

    # Report final distribution
    print(f"\n[{split_name}] Done! {total_base} total frames.")
    print(f"  Zombie count distribution:")
    for c in range(max_zombies + 1):
        print(f"    {c} zombies: {count_tracker.get(c, 0)} frames")


def main():
    parser = argparse.ArgumentParser(description="Generate zombie detection dataset")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "zombie_detection" / "config.yaml"),
        help="Path to config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]

    base_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]

    splits = [
        ("train", ds_cfg["num_train_examples"]),
        ("val", ds_cfg["num_val_examples"]),
        ("test", ds_cfg["num_test_examples"]),
    ]

    os.environ["SDL_VIDEODRIVER"] = "dummy"

    for split_name, num_examples in splits:
        generate_split(
            split_name=split_name,
            num_examples=num_examples,
            output_dir=base_dir / split_name,
            cfg=cfg,
        )

    print("\nDataset generation complete!")
    print(f"Output directory: {base_dir}")


if __name__ == "__main__":
    main()
