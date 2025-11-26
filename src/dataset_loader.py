"""
Load and preprocess CAN datasets for the IDS system.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CANDatasetLoader:
    """Handles loading and preprocessing of CAN bus datasets"""
    
    def __init__(self, data_path: Union[str, Path]):
        self.data_path: Path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_canmap_voltage_dataset(self, dataset_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load voltage traces from CANMAP dataset.
        Expects CSV files with timestamp, can_id, voltage_samples, and label columns.
        """
        if dataset_path is None:
            dataset_path = self.data_path / "canmap_voltage"
        else:
            dataset_path = Path(dataset_path)
            
        logger.info(f"Loading CANMAP voltage dataset from {dataset_path}")
        
        # Check if dataset exists
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            logger.info("Creating sample dataset structure...")
            return self._create_sample_voltage_data()
        
        # Try to load various file formats
        data_frames = []
        
        for file in dataset_path.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                data_frames.append(df)
                logger.info(f"Loaded {file.name}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {file.name}: {e}")
        
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            logger.info(f"Total records loaded: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No data files found, creating sample data")
            return self._create_sample_voltage_data()
    
    def load_road_dataset(self, dataset_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load ROAD CAN IDS dataset with labeled attack messages.
        Expects CSV with timestamp, can_id, dlc, data, and label columns.
        """
        if dataset_path is None:
            dataset_path = self.data_path / "road_can_ids"
        else:
            dataset_path = Path(dataset_path)
            
        logger.info(f"Loading ROAD CAN IDS dataset from {dataset_path}")
        
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            logger.info("Creating sample dataset structure...")
            return self._create_sample_can_data()
        
        data_frames = []
        
        for file in dataset_path.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                data_frames.append(df)
                logger.info(f"Loaded {file.name}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {file.name}: {e}")
        
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            logger.info(f"Total records loaded: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No data files found, creating sample data")
            return self._create_sample_can_data()
    
    def _create_sample_voltage_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate realistic synthetic voltage data based on CAN bus physical layer characteristics.
        
        Based on research from:
        - "Voltage-based Intrusion Detection in Automotive Networks" (ACM 2024)
        - "Physical Layer Fingerprinting for CAN" (IEEE 2024)
        
        Each ECU has unique hardware characteristics that create distinct voltage fingerprints:
        - Rise/fall times (influenced by driver IC and PCB layout)
        - Ringing frequency and damping (LC circuit characteristics)
        - Overshoot/undershoot (impedance mismatches)
        - Settling behavior (capacitive loading)
        """
        logger.info(f"Creating {n_samples} sample voltage records with realistic physical layer characteristics")
        
        data = []
        ecus = [0x100, 0x200, 0x300, 0x400, 0x500]
        
        # Define unique hardware profiles for each ECU based on typical CAN transceiver variations
        ecu_hardware_profiles = {
            0x100: {  # Fast driver, minimal ringing
                'rise_time': 0.15,      # Fast rise time (μs)
                'fall_time': 0.18,
                'ringing_freq': 12.5,   # MHz
                'ringing_damping': 0.8,
                'overshoot': 0.08,      # 8% overshoot
                'undershoot': 0.06,
                'noise_level': 0.02,
                'settling_time': 0.5,
                'capacitance': 45,      # pF - affects signal shape
            },
            0x200: {  # Medium driver, moderate ringing
                'rise_time': 0.22,
                'fall_time': 0.25,
                'ringing_freq': 10.0,
                'ringing_damping': 0.6,
                'overshoot': 0.12,
                'undershoot': 0.09,
                'noise_level': 0.025,
                'settling_time': 0.8,
                'capacitance': 55,
            },
            0x300: {  # Slow driver, more ringing
                'rise_time': 0.30,
                'fall_time': 0.33,
                'ringing_freq': 8.5,
                'ringing_damping': 0.5,
                'overshoot': 0.15,
                'undershoot': 0.12,
                'noise_level': 0.03,
                'settling_time': 1.2,
                'capacitance': 65,
            },
            0x400: {  # Fast with high overshoot
                'rise_time': 0.12,
                'fall_time': 0.14,
                'ringing_freq': 15.0,
                'ringing_damping': 0.7,
                'overshoot': 0.18,
                'undershoot': 0.14,
                'noise_level': 0.022,
                'settling_time': 0.6,
                'capacitance': 40,
            },
            0x500: {  # Moderate with low noise
                'rise_time': 0.20,
                'fall_time': 0.23,
                'ringing_freq': 11.0,
                'ringing_damping': 0.75,
                'overshoot': 0.10,
                'undershoot': 0.08,
                'noise_level': 0.018,
                'settling_time': 0.7,
                'capacitance': 50,
            }
        }
        
        # Sampling parameters
        sample_rate = 1000  # MHz (1 GHz sampling)
        n_voltage_samples = 100  # Number of samples per message
        time_vector = np.linspace(0, n_voltage_samples / sample_rate, n_voltage_samples)
        
        for i in range(n_samples):
            ecu_id = ecus[i % len(ecus)]
            profile = ecu_hardware_profiles[ecu_id]
            
            # Determine if this is an attack (10% attack rate)
            is_attack = np.random.random() < 0.1
            
            if is_attack:
                # Attacker spoofing: uses different hardware, so voltage signature doesn't match
                # Choose a random different ECU profile to simulate attacker hardware
                attacker_ecu = np.random.choice([e for e in ecus if e != ecu_id])
                actual_profile = ecu_hardware_profiles[attacker_ecu]
                attack_type = 'spoofing'
                label = 1
            else:
                actual_profile = profile
                attack_type = 'normal'
                label = 0
            
            # Generate realistic CAN voltage waveform (dominant to recessive transition)
            voltage_samples = self._generate_can_voltage_waveform(
                time_vector, actual_profile, sample_rate
            )
            
            data.append({
                'timestamp': i * 0.01,
                'can_id': ecu_id,
                'ecu_id': ecu_id,
                'voltage_samples': voltage_samples.tolist(),
                'label': label,
                'attack_type': attack_type
            })
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            logger.info(f"Generated voltage data: Normal={np.sum(df['label']==0)}, Attack={np.sum(df['label']==1)}")
        else:
            logger.info("Generated voltage data: Empty dataset")
        return df
    
    def _generate_can_voltage_waveform(self, time_vector: np.ndarray, 
                                       profile: dict, sample_rate: float) -> np.ndarray:
        """
        Generate realistic CAN voltage waveform with hardware-specific characteristics.
        
        CAN uses differential signaling with dominant (2.5V typical) and recessive (0V) states.
        This generates a transition showing the physical layer characteristics.
        """
        n_samples = len(time_vector)
        voltage = np.zeros(n_samples)
        
        # CAN voltage levels
        V_dominant = 2.5  # Dominant state voltage
        V_recessive = 0.0  # Recessive state voltage
        
        # Generate step response with realistic rise time
        rise_samples = int(profile['rise_time'] * sample_rate)
        rise_samples = max(1, min(rise_samples, n_samples // 4))
        
        # Create sigmoid-based rise with hardware-specific characteristics
        for i in range(n_samples):
            if i < rise_samples:
                # Smooth rise with overshoot
                progress = i / rise_samples
                voltage[i] = V_recessive + (V_dominant - V_recessive) * (
                    1 - np.exp(-5 * progress)
                ) * (1 + profile['overshoot'] * np.exp(-3 * progress))
                
                # Add ringing (damped oscillation from LC characteristics)
                ringing_phase = 2 * np.pi * profile['ringing_freq'] * time_vector[i]
                ringing_amplitude = profile['overshoot'] * V_dominant * np.exp(
                    -profile['ringing_damping'] * time_vector[i] * 10
                )
                voltage[i] += ringing_amplitude * np.sin(ringing_phase)
                
            else:
                # Settled to dominant state with small variations
                voltage[i] = V_dominant
        
        # Add thermal and EMI noise (hardware-specific)
        noise = np.random.normal(0, profile['noise_level'] * V_dominant, n_samples)
        voltage += noise
        
        # Add capacitive coupling effects (slight droop over time)
        droop_factor = 1.0 - (time_vector / time_vector[-1]) * (profile['capacitance'] / 1000.0)
        voltage *= droop_factor
        
        # Ensure voltage stays in reasonable range
        voltage = np.clip(voltage, -0.5, 3.5)
        
        return voltage
    
    def _create_sample_can_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create sample CAN message data for testing"""
        logger.info(f"Creating {n_samples} sample CAN records")
        
        # Handle edge cases
        if n_samples <= 0:
            return pd.DataFrame(columns=['timestamp', 'can_id', 'dlc', 'data', 'label', 'attack_type'])
        
        data = []
        can_ids = [0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700]
        attack_types = ['normal', 'dos', 'fuzzing', 'spoofing', 'replay']
        
        # Create bursts of attacks to ensure sequences get labeled as attacks
        for i in range(n_samples):
            can_id = can_ids[i % len(can_ids)]
            
            # Generate CAN data (8 bytes)
            can_data = [np.random.randint(0, 256) for _ in range(8)]
            
            # Create attack bursts - if we're in an attack period, stay in it for a while
            # This ensures sequences will be labeled as attacks
            attack_burst_size = 150  # Length of attack bursts
            if (i // attack_burst_size) % 5 == 0:  # Every 5th burst is an attack
                is_attack = True
                attack_type = attack_types[(i // attack_burst_size) % 4 + 1]  # Cycle through attack types
            else:
                is_attack = False
                attack_type = 'normal'
            
            data.append({
                'timestamp': i * 0.001,
                'can_id': can_id,
                'dlc': 8,
                'data': can_data,
                'label': 1 if is_attack else 0,
                'attack_type': attack_type
            })
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            logger.info(f"Generated CAN data: {len(df)} messages, "
                       f"Normal={np.sum(df['label']==0)}, Attack={np.sum(df['label']==1)}")
        else:
            logger.info("Generated CAN data: Empty dataset")
        return df
    
    def preprocess_voltage_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert raw voltage data into normalized features ready for training"""
        logger.info("Preprocessing voltage data...")
        
        # Extract voltage samples (handle both string and list formats)
        if isinstance(df['voltage_samples'].iloc[0], str):
            voltage_data = df['voltage_samples'].apply(eval).tolist()
        else:
            voltage_data = df['voltage_samples'].tolist()
        
        X = np.array(voltage_data)
        y = df['label'].values
        
        # Normalize each signal to zero mean and unit variance
        X_mean = X.mean(axis=1, keepdims=True)
        X_std = X.std(axis=1, keepdims=True) + 1e-8
        X_normalized = (X - X_mean) / X_std
        
        logger.info(f"Voltage data shape: {X_normalized.shape}")
        logger.info(f"Labels distribution: Normal={np.sum(y==0)}, Attack={np.sum(y==1)}")
        
        return X_normalized, y
    
    def preprocess_can_data(self, df: pd.DataFrame, sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Turn CAN messages into sequences for the deep learning models.
        Each sequence is a sliding window of messages.
        """
        logger.info("Preprocessing CAN message data...")
        
        # Parse data bytes
        if isinstance(df['data'].iloc[0], str):
            data_values = df['data'].apply(eval).tolist()
        else:
            data_values = df['data'].tolist()
        
        # Build feature vectors: [can_id, dlc, 8 data bytes]
        features = []
        for i, row in df.iterrows():
            data_bytes = data_values[i] if isinstance(data_values[i], list) else [0]*8
            feature = [row['can_id']] + [row.get('dlc', 8)] + data_bytes
            features.append(feature)
        
        X = np.array(features)
        
        # Normalize features
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(len(X_normalized) - sequence_length):
            seq = X_normalized[i:i+sequence_length]
            # Use majority label for sequence
            label = df['label'].iloc[i:i+sequence_length].mode()[0]
            sequences.append(seq)
            labels.append(label)
        
        X_seq = np.array(sequences)
        y_seq = np.array(labels)
        
        logger.info(f"Sequences shape: {X_seq.shape}")
        logger.info(f"Labels distribution: Normal={np.sum(y_seq==0)}, Attack={np.sum(y_seq==1)}")
        
        return X_seq, y_seq
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   random_seed: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets using stratified sampling.
        This ensures each split has a balanced representation of both classes.
        
        Args:
            X: Feature array
            y: Label array
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train, val, test splits and their corresponding indices
        """
        np.random.seed(random_seed)
        
        # Stratified split - split each class separately
        normal_indices = np.where(y == 0)[0]
        attack_indices = np.where(y == 1)[0]
        
        # Shuffle each class independently
        np.random.shuffle(normal_indices)
        np.random.shuffle(attack_indices)
        
        # Split each class according to ratios
        n_normal = len(normal_indices)
        n_attack = len(attack_indices)
        
        normal_train_end = int(n_normal * train_ratio)
        normal_val_end = int(n_normal * (train_ratio + val_ratio))
        
        attack_train_end = int(n_attack * train_ratio)
        attack_val_end = int(n_attack * (train_ratio + val_ratio))
        
        # Split normal samples
        normal_train = normal_indices[:normal_train_end]
        normal_val = normal_indices[normal_train_end:normal_val_end]
        normal_test = normal_indices[normal_val_end:]
        
        # Split attack samples
        attack_train = attack_indices[:attack_train_end]
        attack_val = attack_indices[attack_train_end:attack_val_end]
        attack_test = attack_indices[attack_val_end:]
        
        # Combine and shuffle each split
        train_indices = np.concatenate([normal_train, attack_train])
        val_indices = np.concatenate([normal_val, attack_val])
        test_indices = np.concatenate([normal_test, attack_test])
        
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        # Create final splits
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train labels - Normal: {np.sum(y_train==0)}, Attack: {np.sum(y_train==1)}")
        logger.info(f"Val labels - Normal: {np.sum(y_val==0)}, Attack: {np.sum(y_val==1)}")
        logger.info(f"Test labels - Normal: {np.sum(y_test==0)}, Attack: {np.sum(y_test==1)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            # Return indices so caller can track ECU IDs and other metadata
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }


def main():
    """Test dataset loading"""
    loader = CANDatasetLoader("data/raw")
    
    # Load datasets
    voltage_df = loader.load_canmap_voltage_dataset()
    can_df = loader.load_road_dataset()
    
    # Preprocess
    X_voltage, y_voltage = loader.preprocess_voltage_data(voltage_df)
    X_can, y_can = loader.preprocess_can_data(can_df)
    
    # Split
    voltage_splits = loader.split_data(X_voltage, y_voltage)
    can_splits = loader.split_data(X_can, y_can)
    
    logger.info("Dataset loading test complete!")


if __name__ == "__main__":
    main()
