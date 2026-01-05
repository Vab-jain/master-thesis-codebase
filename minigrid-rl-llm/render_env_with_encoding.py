#!/usr/bin/env python3
"""
Script to create a figure with rendered BabyAI environment image 
and its text encoding displayed side by side.

Usage:
    python render_env_with_encoding.py --env BabyAI-GoToObj-v0 --encoding ascii --output figure.png
"""

import os
import sys
import argparse
from typing import Optional

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper
from utils.env import make_env
from utils.observation_encoder import ObservationEncoder


def create_environment_figure(env_id: str = "BabyAI-GoToObj-v0", 
                            encoding_type: str = "ascii",
                            output_path: Optional[str] = None,
                            seed: Optional[int] = None,
                            figsize: tuple = (12, 6)):
    """
    Create a figure showing a BabyAI environment and its text encoding.
    
    Args:
        env_id: Environment ID (e.g., "BabyAI-GoToObj-v0")
        encoding_type: Type of encoding ("natural", "ascii", "tuples", "relative")
        output_path: Path to save the figure (optional)
        seed: Random seed for environment (optional)
        figsize: Figure size as (width, height)
    
    Returns:
        matplotlib figure object
    """
    
    # Create environment with FullyObsWrapper for complete observation
    print(f"🚀 Creating environment: {env_id}")
    env = make_env(env_id, seed=seed, render_mode="rgb_array", wrappers=[FullyObsWrapper])
    
    # Initialize observation encoder
    encoder = ObservationEncoder()
    
    # Reset environment to get initial observation
    obs, info = env.reset()
    
    # Get rendered image
    img = env.render()
    
    if img is None:
        raise ValueError("Could not render environment image")
    
    # Create figure with subplots - more compact layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1.5], 'wspace': 0.2})
    
    # Left subplot: Environment image (smaller)
    # Set aspect ratio to maintain image proportions and align to top
    ax1.imshow(img, aspect='equal')
    ax1.set_title(f"Environment: {env_id}", fontsize=12, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # Ensure image is aligned to top of subplot
    ax1.set_anchor('N')  # North anchor - align to top
    
    # Right subplot: Text encoding (larger)
    ax2.axis('off')
    
    # Get encoded text representation
    try:
        encoded_text = encoder.encode_all(obs)[encoding_type]
    except KeyError:
        available_types = list(encoder.encode_all(obs).keys())
        raise ValueError(f"Invalid encoding type '{encoding_type}'. Available types: {available_types}")
    
    # Format text for display
    display_text = f"ENCODING TYPE: {encoding_type.upper()}\n"
    display_text += "=" * 50 + "\n\n"
    display_text += encoded_text
    
    # Add text to right subplot with larger, more readable formatting
    ax2.text(0.05, 0.95, display_text, 
             transform=ax2.transAxes, 
             fontsize=13,  # Increased for better readability
             verticalalignment='top', 
             horizontalalignment='left',
             fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"),
             wrap=True)
    
    ax2.set_title(f"Text Encoding: {encoding_type.capitalize()}", fontsize=12, fontweight='bold', pad=10)
    
    # Ensure text area is aligned to top of subplot
    ax2.set_anchor('N')  # North anchor - align to top
    
    # Use constrained layout for better spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.2)
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Figure saved to: {output_path}")
    
    # Close environment
    env.close()
    
    return fig


def main():
    """Main function with command line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Render BabyAI environment with text encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python render_env_with_encoding.py --env BabyAI-GoToObj-v0 --encoding ascii
  python render_env_with_encoding.py --env BabyAI-OpenDoor-v0 --encoding natural --output door_env.png
  python render_env_with_encoding.py --env BabyAI-PickupLoc-v0 --encoding tuples --seed 42
        """
    )
    
    parser.add_argument("--env", "-e", 
                       default="BabyAI-GoToObj-v0",
                       help="Environment ID (default: BabyAI-GoToObj-v0)")
    
    parser.add_argument("--encoding", "-enc",
                       default="ascii", 
                       choices=["natural", "ascii", "tuples", "relative"],
                       help="Text encoding type (default: ascii)")
    
    parser.add_argument("--output", "-o",
                       help="Output file path (optional)")
    
    parser.add_argument("--seed", "-s",
                       type=int,
                       help="Random seed for environment (optional)")
    
    parser.add_argument("--figsize",
                       nargs=2, type=float, default=[12, 6],
                       help="Figure size as width height (default: 12 6)")
    
    parser.add_argument("--show", 
                       action="store_true",
                       help="Show the figure interactively (blocks execution)")
    
    args = parser.parse_args()
    
    print(f"Creating figure for environment '{args.env}' with '{args.encoding}' encoding...")
    
    try:
        # Create the figure
        fig = create_environment_figure(
            env_id=args.env,
            encoding_type=args.encoding,
            output_path=args.output,
            seed=args.seed,
            figsize=tuple(args.figsize)
        )
        
        # Show figure if requested
        if args.show:
            plt.show()
        elif not args.output:
            # If no output specified and not showing, display briefly then close
            plt.show(block=False)
            plt.pause(2)  # Show for 2 seconds
            plt.close(fig)
            print("Figure displayed for 2 seconds. Use --show to keep it open or --output to save.")
        
        print("✅ Script completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 