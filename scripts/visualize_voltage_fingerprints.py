"""
Visualize the improved voltage fingerprinting - show different ECU waveforms
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import CANDatasetLoader

def visualize_voltage_waveforms():
    """Visualize voltage waveforms from different ECUs"""
    
    print("\nGenerating voltage waveform visualization...")
    
    # Generate samples
    loader = CANDatasetLoader(data_path='.')
    voltage_df = loader._create_sample_voltage_data(n_samples=100)
    
    # Get one sample from each ECU
    ecus = [0x100, 0x200, 0x300, 0x400, 0x500]
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 12))
    fig.suptitle('CAN Bus Voltage Fingerprinting - Hardware-Specific Waveforms', 
                 fontsize=16, fontweight='bold')
    
    for idx, ecu_id in enumerate(ecus):
        # Get normal and attack samples for this ECU
        ecu_samples = voltage_df[voltage_df['ecu_id'] == ecu_id]
        normal_samples = ecu_samples[ecu_samples['label'] == 0]
        attack_samples = ecu_samples[ecu_samples['label'] == 1]
        
        if len(normal_samples) > 0:
            normal_signal = np.array(normal_samples.iloc[0]['voltage_samples'])
            time = np.arange(len(normal_signal))
            
            # Plot normal waveform
            axes[idx, 0].plot(time, normal_signal, 'g-', linewidth=1.5, alpha=0.8)
            axes[idx, 0].set_title(f'ECU 0x{ecu_id:03X} - Genuine Message', fontweight='bold')
            axes[idx, 0].set_ylabel('Voltage (V)')
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].set_ylim(-0.5, 3.5)
            
            # Annotate key features
            rise_time = np.where(normal_signal > 1.5)[0]
            if len(rise_time) > 0:
                rise_idx = rise_time[0]
                axes[idx, 0].axvline(rise_idx, color='blue', linestyle='--', 
                                    alpha=0.5, label='Rise time')
                axes[idx, 0].annotate('Rise', xy=(rise_idx, 1.5), 
                                     xytext=(rise_idx+10, 1.0),
                                     arrowprops=dict(arrowstyle='->', color='blue'),
                                     fontsize=9)
            
            # Mark overshoot
            peak_idx = np.argmax(normal_signal)
            if normal_signal[peak_idx] > 2.6:
                axes[idx, 0].plot(peak_idx, normal_signal[peak_idx], 'r*', 
                                markersize=12, label='Overshoot')
        
        if len(attack_samples) > 0:
            attack_signal = np.array(attack_samples.iloc[0]['voltage_samples'])
            time = np.arange(len(attack_signal))
            
            # Plot attack waveform (different hardware)
            axes[idx, 1].plot(time, attack_signal, 'r-', linewidth=1.5, alpha=0.8)
            axes[idx, 1].set_title(f'ECU 0x{ecu_id:03X} - Spoofed Message (Different HW)', 
                                  fontweight='bold', color='darkred')
            axes[idx, 1].set_ylabel('Voltage (V)')
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].set_ylim(-0.5, 3.5)
            
            # Highlight differences
            if len(normal_samples) > 0:
                diff = np.abs(attack_signal - normal_signal)
                axes[idx, 1].fill_between(time, attack_signal - diff/2, 
                                         attack_signal + diff/2,
                                         alpha=0.3, color='yellow', 
                                         label='Signature mismatch')
        else:
            # No attack sample, just show normal again for reference
            axes[idx, 1].plot(time, normal_signal, 'g-', linewidth=1.5, alpha=0.8)
            axes[idx, 1].set_title(f'ECU 0x{ecu_id:03X} - No Attack Sample',
                                  fontweight='bold')
            axes[idx, 1].set_ylabel('Voltage (V)')
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].set_ylim(-0.5, 3.5)
        
        # Only show x-label on bottom row
        if idx == 4:
            axes[idx, 0].set_xlabel('Time (samples)', fontweight='bold')
            axes[idx, 1].set_xlabel('Time (samples)', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = 'visualization_output/voltage_waveforms.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Voltage waveform visualization saved to: {output_path}")
    
    # Create a comparison plot showing all ECUs together
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('ECU Hardware Signatures - Physical Layer Characteristics', 
                  fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for idx, ecu_id in enumerate(ecus):
        ecu_samples = voltage_df[voltage_df['ecu_id'] == ecu_id]
        normal_samples = ecu_samples[ecu_samples['label'] == 0]
        
        if len(normal_samples) > 0:
            signal = np.array(normal_samples.iloc[0]['voltage_samples'])
            time = np.arange(len(signal))
            
            ax1.plot(time, signal, color=colors[idx], linewidth=2, 
                    alpha=0.7, label=f'ECU 0x{ecu_id:03X}')
    
    ax1.set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')
    ax1.set_title('Unique Hardware Signatures per ECU', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 3.5)
    
    # Show feature comparison
    features_to_plot = []
    ecu_labels = []
    
    for ecu_id in ecus:
        ecu_samples = voltage_df[voltage_df['ecu_id'] == ecu_id]
        normal_samples = ecu_samples[ecu_samples['label'] == 0]
        
        if len(normal_samples) > 0:
            signal = np.array(normal_samples.iloc[0]['voltage_samples'])
            
            # Calculate key features
            rise_samples = np.where(signal > 1.5)[0]
            rise_time = rise_samples[0] if len(rise_samples) > 0 else 0
            overshoot = np.max(signal) - 2.5
            settling = np.mean(signal[-20:])
            
            features_to_plot.append([rise_time, overshoot*100, settling])
            ecu_labels.append(f'0x{ecu_id:03X}')
    
    features_array = np.array(features_to_plot)
    x = np.arange(len(ecu_labels))
    width = 0.25
    
    ax2.bar(x - width, features_array[:, 0], width, label='Rise Time (samples)', 
            color='steelblue', alpha=0.8)
    ax2.bar(x, features_array[:, 1], width, label='Overshoot (%)', 
            color='orange', alpha=0.8)
    ax2.bar(x + width, features_array[:, 2]*100, width, label='Settling (V×100)', 
            color='green', alpha=0.8)
    
    ax2.set_xlabel('ECU ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
    ax2.set_title('Distinctive Physical Layer Features', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ecu_labels)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path2 = 'visualization_output/voltage_signatures_comparison.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Signature comparison saved to: {output_path2}")
    
    print("\n" + "="*70)
    print("Voltage Fingerprinting Visualization Complete!")
    print("="*70)
    print("\nKey Observations:")
    print("• Each ECU has unique hardware characteristics")
    print("• Rise times, overshoot, and ringing patterns differ")
    print("• Physical layer signatures enable spoofing detection")
    print("• Attacker using different hardware creates detectable anomalies")
    print("="*70 + "\n")


if __name__ == "__main__":
    visualize_voltage_waveforms()
