#!/usr/bin/env python3
"""
Export a single MiniGrid/BabyAI observation:
- Saves the true environment-rendered image (PNG)
- Writes 4 complete prompt files (ASCII, RELATIVE, TUPLES, NATURAL)
  using the SAME observation and a shared task_description.

Usage:
  python export_single_observation.py --env BabyAI-OpenDoor-v0 --seed 42 --outdir LLM_prompt_export
"""

import os
import sys
import argparse
from typing import Optional, Dict

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: F401  (imported for side effects in wrappers/env creation)
from minigrid.wrappers import FullyObsWrapper  # type: ignore
from utils.env import make_env  # type: ignore
from utils.observation_encoder import ObservationEncoder  # type: ignore
from utils.config_task_desc import task_desc  # type: ignore

import matplotlib.pyplot as plt


def save_png(image_array, output_path: str) -> None:
    """
    Save an RGB numpy array as PNG.
    """
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(image_array)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def write_prompt_file(path: str, task_description: str, current_state: str) -> None:
    with open(path, "w") as f:
        f.write("task_description:\n")
        f.write(task_description + "\n\n")
        f.write("current_state:\n")
        f.write(current_state + "\n\n")
        f.write("previous_actions:\n\n")


def main():
    parser = argparse.ArgumentParser(description="Export one observation with true MiniGrid render and per-encoding prompts.")
    parser.add_argument("--env", default="BabyAI-OpenDoor-v0", help="Environment ID (e.g., BabyAI-OpenDoor-v0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument("--outdir", default="LLM_prompt_export", help="Output directory to write image and prompts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Create environment with full observation and render support
    env = make_env(env_key=args.env, seed=args.seed, render_mode="rgb_array", wrappers=[FullyObsWrapper])
    obs, _ = env.reset()

    # Render true environment image
    img = env.render()
    if img is None:
        raise RuntimeError("env.render() returned None; ensure render_mode='rgb_array'.")

    # Encode current observation into four textual encodings
    encoder = ObservationEncoder()
    encodings: Dict[str, str] = encoder.encode_all(obs)

    # Use shared task description
    td = task_desc

    # Save image
    image_path = os.path.join(args.outdir, "observation.png")
    save_png(img, image_path)

    # Save 4 separate prompt files (same observation, different encodings)
    prompts = {
        "ascii": encodings.get("ascii", ""),
        "relative": encodings.get("relative", ""),
        "tuples": encodings.get("tuples", ""),
        "natural": encodings.get("natural", ""),
    }

    write_prompt_file(os.path.join(args.outdir, "LLM_prompt_ascii.txt"), td, f"(ASCII)\n{prompts['ascii']}")
    write_prompt_file(os.path.join(args.outdir, "LLM_prompt_relative.txt"), td, f"(RELATIVE)\n{prompts['relative']}")
    write_prompt_file(os.path.join(args.outdir, "LLM_prompt_tuples.txt"), td, f"(TUPLES)\n{prompts['tuples']}")
    write_prompt_file(os.path.join(args.outdir, "LLM_prompt_natural.txt"), td, f"(NATURAL)\n{prompts['natural']}")

    # Also save a summary file listing the mission if present
    mission = obs.get("mission", "N/A")
    with open(os.path.join(args.outdir, "README.txt"), "w") as f:
        f.write(f"Environment: {args.env}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Mission: {mission}\n")
        f.write(f"Image: observation.png\n")
        f.write("Prompts:\n")
        f.write("  - LLM_prompt_ascii.txt\n")
        f.write("  - LLM_prompt_relative.txt\n")
        f.write("  - LLM_prompt_tuples.txt\n")
        f.write("  - LLM_prompt_natural.txt\n")

    env.close()
    print(f"Saved image and prompts to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()






