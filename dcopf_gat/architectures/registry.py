from __future__ import annotations
from typing import Dict, Type
from .base import Architecture

ARCH_REGISTRY: Dict[str, Type[Architecture]] = {}

def register(name: str):
    def deco(cls):
        cls.name = name
        ARCH_REGISTRY[name] = cls
        return cls
    return deco
