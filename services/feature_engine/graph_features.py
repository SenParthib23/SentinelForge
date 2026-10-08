"""
SentinelForge — services/feature_engine/graph_features.py

PURPOSE:
    Constructs in-memory bipartite identity graphs connecting Customers to
    shared Device Fingerprints and IP Subnets to uncover fraud syndicates
    and linked mule accounts.

ARCHITECTURE POSITION:
    Layer 2 (Feature Engine) & Layer 4 (Investigation Case Context).

INTERVIEW TALKING POINT:
    "Organized fraud rings frequently recycle physical devices and VPN endpoints
    across multiple synthetic or hijacked accounts. Our graph feature module
    maintains a bipartite customer-device-IP association index, immediately identifying
    if a newly registered account shares hardware with known fraudulent accounts."
"""

from collections import defaultdict
from typing import Optional, Set


class GraphFeatureComputer:
    """
    Bipartite entity graph tracking shared hardware and network identifiers.
    """

    def __init__(self) -> None:
        """Initialize graph adjacency index."""
        # device_fingerprint -> set of customer_ids
        self._device_to_customers: dict[str, set[str]] = defaultdict(set)
        # ip_address -> set of customer_ids
        self._ip_to_customers: dict[str, set[str]] = defaultdict(set)
        # customer_id -> set of device_fingerprints
        self._customer_to_devices: dict[str, set[str]] = defaultdict(set)

    def record_interaction(
        self,
        customer_id: str,
        device_fingerprint: str,
        ip_address: Optional[str] = None,
    ) -> None:
        """
        Add edge between customer, device, and IP in the identity graph.

        Args:
            customer_id: Customer ID.
            device_fingerprint: Device fingerprint hash.
            ip_address: Optional client IP.
        """
        self._device_to_customers[device_fingerprint].add(customer_id)
        self._customer_to_devices[customer_id].add(device_fingerprint)
        if ip_address:
            self._ip_to_customers[ip_address].add(customer_id)

    def get_linked_accounts(
        self,
        customer_id: str,
        device_fingerprint: str,
        ip_address: Optional[str] = None,
    ) -> set[str]:
        """
        Find all other customer accounts that share device or IP with this customer.

        Args:
            customer_id: Target customer ID.
            device_fingerprint: Device fingerprint.
            ip_address: Optional IP.

        Returns:
            set[str]: Set of distinct linked customer_ids (excluding the target customer).
        """
        linked: set[str] = set()

        # Shared device links
        if device_fingerprint in self._device_to_customers:
            linked.update(self._device_to_customers[device_fingerprint])

        # Shared IP links
        if ip_address and ip_address in self._ip_to_customers:
            linked.update(self._ip_to_customers[ip_address])

        # Exclude self
        linked.discard(customer_id)
        return linked
