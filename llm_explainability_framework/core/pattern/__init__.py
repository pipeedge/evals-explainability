"""
Pattern Library Module

This module provides comprehensive pattern library capabilities for
failure pattern collection, validation, and management in the MADF framework.
"""

from .pattern_library_system import (
    PatternLibrarySystem, 
    FailurePattern, 
    PatternExample, 
    CounterfactualFix,
    PatternSeverity,
    PatternSource
)

__all__ = [
    'PatternLibrarySystem',
    'FailurePattern',
    'PatternExample', 
    'CounterfactualFix',
    'PatternSeverity',
    'PatternSource'
] 