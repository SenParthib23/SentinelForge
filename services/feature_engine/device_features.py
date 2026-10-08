"""
SentinelForge — services/feature_engine/device_features.py

PURPOSE:
    Computes device fingerprint diversity, novelty indicators, and shared device
    velocity across accounts to detect account takeover and credential stuffing.

ARCHITECTURE POSITION:
    Layer 2 (Feature Engine) → Device Subsystem.

FEATURES COMPUTED:
    - unique_devices_1hr: Distinct devices seen for customer in last 1 hour.
    - device_txn_count: Global transaction count for this device in last 1 hour.
    - is_new_device: True if device fingerprint is novel for this customer in 7 days.
"""

from collections import defaultdict
from typing import Optional


class DeviceFeatureComputer:
    """
    Tracks device fingerprints and computes novelty & diversity metrics.
    """

    def __init__(self) -> None:
        """Initialize in-memory device history tables."""
        # customer_id -> list of (device_fingerprint, timestamp_epoch)
        self._customer_device_history: dict[str, list[tuple[str, float]]] = defaultdict(list)
        # customer_id -> set of all historical known device fingerprints
        self._customer_known_devices: dict[str, set[str]] = defaultdict(set)
        # device_fingerprint -> list of timestamp_epochs
        self._device_global_history: dict[str, list[float]] = defaultdict(list)

    def register_known_device(self, customer_id: str, device_fingerprint: str) -> None:
        """Pre-seed customer's historical primary device."""
        self._customer_known_devices[customer_id].add(device_fingerprint)

    def compute_device_features(
        self,
        customer_id: str,
        device_fingerprint: str,
        timestamp_epoch: float,
    ) -> dict[str, any]:
        """
        Record device interaction and compute device novelty and diversity features.

        Args:
            customer_id: Customer ID.
            device_fingerprint: Device hardware/browser hash.
            timestamp_epoch: Unix epoch seconds.

        Returns:
            dict[str, any]: Computed device features:
                - unique_devices_1hr: int
                - device_txn_count: int
                - is_new_device: bool
        """
        # 1. Determine if device is novel for this customer
        is_new = device_fingerprint not in self._customer_known_devices[customer_id]
        self._customer_known_devices[customer_id].add(device_fingerprint)

        # 2. Record device for customer
        self._customer_device_history[customer_id].append((device_fingerprint, timestamp_epoch))
        self._device_global_history[device_fingerprint].append(timestamp_epoch)

        # 3. Prune entries older than 1 hour (3600s)
        cutoff_1hr = timestamp_epoch - 3600
        self._customer_device_history[customer_id] = [
            (dev, ts)
            for dev, ts in self._customer_device_history[customer_id]
            if ts >= cutoff_1hr
        ]
        self._device_global_history[device_fingerprint] = [
            ts
            for ts in self._device_global_history[device_fingerprint]
            if ts >= cutoff_1hr
        ]

        # 4. Count unique devices for customer in 1hr
        unique_devs_1hr = len(
            {dev for dev, ts in self._customer_device_history[customer_id]}
        )
        dev_txn_count_1hr = len(self._device_global_history[device_fingerprint])

        return {
            "unique_devices_1hr": max(1, unique_devs_1hr),
            "device_txn_count": max(1, dev_txn_count_1hr),
            "is_new_device": is_new,
        }
