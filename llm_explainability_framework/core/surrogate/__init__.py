"""
Surrogate Model Integration Module

This module provides surrogate model selection and integration capabilities
for improving cross-attention approximation in the MADF framework.
"""

from .surrogate_model_selector import (
    SurrogateModelSelector, 
    SelectionCriteria, 
    SurrogateModel,
    TaskComplexity,
    DomainType
)
from .integrate_surrogate_models import (
    EnhancedMADFFramework,
    SurrogateModelIntegration
)

__all__ = [
    'SurrogateModelSelector',
    'SelectionCriteria', 
    'SurrogateModel',
    'TaskComplexity',
    'DomainType',
    'EnhancedMADFFramework',
    'SurrogateModelIntegration'
] 