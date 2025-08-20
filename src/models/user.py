"""
User data models for WordOfPrompt.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UserModel:
    """User model for WordOfPrompt system."""
    
    id: str
    email: str
    tier: str = "basic"
    balance: float = 0.0
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    preferences: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "email": self.email,
            "tier": self.tier,
            "balance": self.balance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "preferences": self.preferences or {}
        }
