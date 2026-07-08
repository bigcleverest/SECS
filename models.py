"""
Data Models for Fire Alarm System
消防报警系统数据模型

Contains all data structures, enums, and dataclasses used throughout the system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


# ============== ENUMERATIONS ==============

class EventType(Enum):
    """Event type enumeration based on emergency categories"""
    BUILDING_FIRE = "建筑火灾"
    FOREST_FIRE = "林草燃烧"
    VEHICLE_FIRE = "车辆火灾"
    GAS_LEAK = "气体泄露"
    HIGH_ALTITUDE_RESCUE = "高空／深井被困救援"
    ELEVATOR_RESCUE = "电梯被困"
    FLOOD_DRAINAGE = "洪涝排涝"
    WATER_RESCUE = "水域救援"
    WILDLIFE_HANDLING = "野生动物处理"
    BURGLARY = "入室盗窃"
    MISSING_CHILD = "儿童走失"
    MISSING_PET = "宠物走失"
    NOISE_DISTURBANCE = "噪音扰民"
    POWER_WATER_OUTAGE = "停电停水"


class EventState(Enum):
    """Event state progression following Markov process"""
    INITIAL = "初始态"
    SMOKE = "冒烟"
    OPEN_FLAME = "明火"
    FLASHOVER = "轰燃"
    CONTROLLED = "受控"
    EXTINGUISHED = "扑灭"


class CallerRole(Enum):
    """Types of emergency callers"""
    VICTIM = "当事人/被困者"
    BYSTANDER = "路人/旁观者"
    REPRESENTATIVE = "群众代表/报警代表"


# ============== DATA STRUCTURES ==============

@dataclass
class Location:
    """Geographic location with address details"""
    address: str
    coordinates: Tuple[float, float]  # (lat, lon)
    landmark: Optional[str] = None

@dataclass
class SituationUpdate:
    """Represents a situation update at a specific time"""
    time: float
    situation: str
    visibility: float  # How visible/obvious the event is [0-1]
    impact: float      # How many people are affected [0-1]
    severity: float    # How severe the situation is [0-1]


@dataclass
class Event:
    """Core event data structure"""
    event_id: str
    event_type: EventType
    start_time: float
    location: str
    current_state: EventState = EventState.INITIAL
    situation_history: List[SituationUpdate] = field(default_factory=list)
    next_state_time: Optional[float] = None


@dataclass
class AlarmCall:
    """Represents an alarm call made by a caller"""
    call_id: str
    event_id: str
    caller_role: CallerRole
    call_time: float
    description: str
    uncertainty_level: float  # [0-1] higher means more uncertain
    emotional_level: float    # [0-1] higher means more emotional


@dataclass
class TimelineEntry:
    """Entry in the simulation timeline"""
    time: float
    entry_type: str  # "event_start", "state_change", "alarm_call"
    data: dict