"""
Fire Alarm System MVP - Emergency Response Simulation
消防报警系统MVP - 应急响应仿真

A comprehensive simulation system for emergency response scenarios
using large language models to generate realistic alarm calls and situations.
"""

# Core components
from .models import (
    EventType, EventState, CallerRole,
    Event, AlarmCall, SituationUpdate, TimelineEntry
)

from .timeline import ClockTimeline
from .llm_interface import LLMInterface, MockLLMInterface
from .state_machine import StateMachine
from .event_factory import EventFactory
from .caller_farm import CallerFarm
from .simulation import FireAlarmSimulation

# Version info
__version__ = "1.0.0"
__author__ = "Fire Alarm System Team"
__description__ = "基于大模型智能体构建消防报警人角色的最小可运行版本"

# Public API
__all__ = [
    # Data models
    'EventType', 'EventState', 'CallerRole',
    'Event', 'AlarmCall', 'SituationUpdate', 'TimelineEntry',
    
    # Core components
    'ClockTimeline',
    'LLMInterface', 'MockLLMInterface',
    'StateMachine',
    'EventFactory',
    'CallerFarm',
    'FireAlarmSimulation',
    
    # Metadata
    '__version__', '__author__', '__description__'
]

# Quick start helper function
def create_simulation(time_limit=60.0, use_mock_llm=True, **kwargs):
    """
    Quick start function to create a simulation with default settings
    
    Args:
        time_limit: Simulation time limit
        use_mock_llm: Whether to use mock LLM interface
        **kwargs: Additional parameters for simulation
        
    Returns:
        Configured FireAlarmSimulation instance
    """
    llm = MockLLMInterface() if use_mock_llm else LLMInterface()
    simulation = FireAlarmSimulation(time_limit, llm)
    
    # Apply any additional parameters
    if kwargs:
        simulation.set_parameters(**kwargs)
    
    return simulation