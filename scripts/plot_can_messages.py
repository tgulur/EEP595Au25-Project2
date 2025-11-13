"""
Visualize CAN bus message traffic over time
Shows message patterns, ECU activity, and attack/normal periods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.dataset_loader import CANDatasetLoader

# Set up nice plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def plot_can_bus_timeline(n_messages=2000, save_path='can_bus_timeline.png'):
    """
    Create comprehensive visualization of CAN bus message traffic
    
    Args:
        n_messages: Number of CAN messages to generate and plot
        save_path: Where to save the plot
    """
    print(f"\n{'='*60}")
    print("CAN Bus Message Timeline Visualization")
    print(f"{'='*60}\n")
    
    # Generate sample CAN data using the loader
    print(f"Generating {n_messages} CAN messages...")
    loader = CANDatasetLoader(data_path='.')  # Path doesn't matter for sample data
    can_df = loader._create_sample_can_data(n_messages)
    
    # Convert dataframe to numpy array format
    timestamps = can_df['timestamp'].values
    can_ids = can_df['can_id'].values
    data_bytes = np.array([msg for msg in can_df['data'].values])  # 8 data bytes
    labels = can_df['label'].values
    
    # Normalize timestamps to start at 0
    timestamps = timestamps - timestamps[0]
    
    # Identify unique CAN IDs (ECUs)
    unique_ids = np.unique(can_ids)
    print(f"Found {len(unique_ids)} unique CAN IDs (ECUs)")
    
    # Count messages per ID
    id_counts = {int(can_id): np.sum(can_ids == can_id) for can_id in unique_ids}
    print("\nMessage distribution by CAN ID:")
    for can_id, count in sorted(id_counts.items()):
        print(f"  ID 0x{can_id:03X}: {count} messages")
    
    # Attack statistics
    n_attacks = np.sum(labels == 1)
    n_normal = np.sum(labels == 0)
    print(f"\nMessage classification:")
    print(f"  Normal: {n_normal} ({n_normal/len(labels)*100:.1f}%)")
    print(f"  Attack: {n_attacks} ({n_attacks/len(labels)*100:.1f}%)")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Message timeline by CAN ID (main visualization)
    ax1 = fig.add_subplot(gs[0:2, :])
    
    # Color by attack/normal
    colors = ['green' if label == 0 else 'red' for label in labels]
    
    # Plot each message as a point
    for i, (t, can_id, label) in enumerate(zip(timestamps, can_ids, labels)):
        color = 'green' if label == 0 else 'red'
        ax1.scatter(t, can_id, c=color, s=20, alpha=0.6, edgecolors='none')
    
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CAN ID (Hex)', fontsize=12, fontweight='bold')
    ax1.set_title('CAN Bus Message Timeline - All ECUs', fontsize=14, fontweight='bold')
    
    # Set y-axis to show CAN IDs in hex
    y_ticks = sorted(unique_ids)
    y_labels = [f"0x{int(y):03X}" for y in y_ticks]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Normal Messages'),
        Patch(facecolor='red', alpha=0.6, label='Attack Messages')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Plot 2: Message rate over time
    ax2 = fig.add_subplot(gs[2, 0])
    
    # Bin messages into time windows
    time_bins = np.arange(0, timestamps[-1], 1.0)  # 1 second bins
    hist_all, _ = np.histogram(timestamps, bins=time_bins)
    hist_normal, _ = np.histogram(timestamps[labels == 0], bins=time_bins)
    hist_attack, _ = np.histogram(timestamps[labels == 1], bins=time_bins)
    
    bin_centers = time_bins[:-1] + 0.5
    ax2.plot(bin_centers, hist_all, 'b-', linewidth=2, label='Total Messages', alpha=0.7)
    ax2.fill_between(bin_centers, hist_normal, alpha=0.5, color='green', label='Normal')
    ax2.fill_between(bin_centers, hist_attack, alpha=0.5, color='red', label='Attacks')
    
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Messages per Second', fontsize=11, fontweight='bold')
    ax2.set_title('Message Rate Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Message distribution by CAN ID
    ax3 = fig.add_subplot(gs[2, 1])
    
    can_id_labels = [f"0x{int(cid):03X}" for cid in sorted(unique_ids)]
    message_counts = [id_counts[int(cid)] for cid in sorted(unique_ids)]
    
    bars = ax3.bar(range(len(unique_ids)), message_counts, color='steelblue', alpha=0.7)
    ax3.set_xlabel('CAN ID', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Messages', fontsize=11, fontweight='bold')
    ax3.set_title('Message Count by CAN ID', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(unique_ids)))
    ax3.set_xticklabels(can_id_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Attack periods visualization
    ax4 = fig.add_subplot(gs[3, :])
    
    # Show attack periods as filled regions
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) > 0:
        # Find continuous attack periods
        attack_periods = []
        period_start = attack_indices[0]
        
        for i in range(1, len(attack_indices)):
            if attack_indices[i] - attack_indices[i-1] > 1:
                # Gap found, end current period
                attack_periods.append((timestamps[period_start], timestamps[attack_indices[i-1]]))
                period_start = attack_indices[i]
        
        # Add last period
        attack_periods.append((timestamps[period_start], timestamps[attack_indices[-1]]))
        
        print(f"\nDetected {len(attack_periods)} attack burst(s):")
        for i, (start, end) in enumerate(attack_periods):
            duration = end - start
            print(f"  Burst {i+1}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
            ax4.axvspan(start, end, alpha=0.3, color='red')
    
    # Plot binary timeline
    ax4.fill_between(timestamps, 0, labels, step='mid', alpha=0.5, color='red', label='Attack')
    ax4.fill_between(timestamps, 0, 1-labels, step='mid', alpha=0.5, color='green', label='Normal')
    
    ax4.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Traffic Type', fontsize=11, fontweight='bold')
    ax4.set_title('Attack/Normal Timeline', fontsize=12, fontweight='bold')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Normal', 'Attack'])
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.legend(loc='upper right')
    
    # Add overall title
    fig.suptitle(f'CAN Bus Network Traffic Analysis - {n_messages} Messages', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {save_path}")
    
    # Also show it
    plt.show()
    
    return fig, can_df


def plot_data_byte_heatmap(n_messages=1000, save_path='can_data_heatmap.png'):
    """
    Create a heatmap showing the data byte values over time
    
    Args:
        n_messages: Number of messages to visualize
        save_path: Where to save the plot
    """
    print(f"\n{'='*60}")
    print("CAN Bus Data Byte Heatmap")
    print(f"{'='*60}\n")
    
    # Generate data using the loader
    print(f"Generating {n_messages} CAN messages...")
    loader = CANDatasetLoader(data_path='.')  # Path doesn't matter for sample data
    can_df = loader._create_sample_can_data(n_messages)
    
    timestamps = can_df['timestamp'].values
    timestamps = timestamps - timestamps[0]  # Normalize to start at 0
    data_bytes = np.array([msg for msg in can_df['data'].values])  # 8 data bytes
    labels = can_df['label'].values
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[4, 1])
    
    # Plot heatmap of data bytes
    im = ax1.imshow(data_bytes.T, aspect='auto', cmap='viridis', 
                    interpolation='nearest', extent=[0, timestamps[-1], 0, 8])
    
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Data Byte Index', fontsize=12, fontweight='bold')
    ax1.set_title('CAN Message Data Byte Values Over Time', fontsize=14, fontweight='bold')
    ax1.set_yticks(np.arange(8) + 0.5)
    ax1.set_yticklabels([f'Byte {i}' for i in range(8)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label='Byte Value (0-255)')
    
    # Add attack/normal timeline at bottom
    colors_timeline = ['green' if l == 0 else 'red' for l in labels]
    ax2.scatter(timestamps, np.zeros_like(timestamps), c=colors_timeline, 
                s=10, marker='|', alpha=0.6)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Type', fontsize=11, fontweight='bold')
    ax2.set_yticks([0])
    ax2.set_yticklabels(['Attack/Normal'])
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Normal'),
        Patch(facecolor='red', alpha=0.6, label='Attack')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {save_path}")
    plt.show()
    
    return fig


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CAN Bus Traffic Visualization Tool")
    print("="*60)
    
    # Create output directory
    output_dir = Path('visualization_output')
    output_dir.mkdir(exist_ok=True)
    
    # Generate timeline plot with 2000 messages
    print("\n[1/2] Creating CAN bus timeline visualization...")
    fig1, can_df = plot_can_bus_timeline(
        n_messages=2000,
        save_path=str(output_dir / 'can_bus_timeline.png')
    )
    
    # Generate data byte heatmap with 1000 messages
    print("\n[2/2] Creating data byte heatmap...")
    fig2 = plot_data_byte_heatmap(
        n_messages=1000,
        save_path=str(output_dir / 'can_data_heatmap.png')
    )
    
    print("\n" + "="*60)
    print("✅ All visualizations complete!")
    print(f"📁 Outputs saved to: {output_dir.absolute()}")
    print("="*60 + "\n")
