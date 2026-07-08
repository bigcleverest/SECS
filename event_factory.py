"""
Event Factory for Fire Alarm System
消防报警系统事件工厂

Factory for creating and managing emergency events using Poisson process.
Generates events with LLM-generated descriptions and manages their lifecycle.
"""

import numpy as np
from typing import Dict
from models import Event, EventType, EventState, SituationUpdate
from timeline import ClockTimeline
from llm_interface import LLMInterface
from state_machine import StateMachine
from ld_hawks_simulation import get_simulated_ts  


class EventFactory:
    """Factory for creating and managing emergency events"""

    EVENT_TYPE_PRIOR_COUNTS: Dict[EventType, int] = {
        EventType.BUILDING_FIRE: 362896,
        EventType.FOREST_FIRE: 10845,
        EventType.VEHICLE_FIRE: 74697,
        EventType.GAS_LEAK: 54773,
        EventType.HIGH_ALTITUDE_RESCUE: 29493,
        EventType.ELEVATOR_RESCUE: 64080,
        EventType.FLOOD_DRAINAGE: 79629,
        EventType.WATER_RESCUE: 4839,
        EventType.WILDLIFE_HANDLING: 9243,
        EventType.BURGLARY: 80119,
        EventType.MISSING_CHILD: 11217,
        EventType.MISSING_PET: 2985,
        EventType.NOISE_DISTURBANCE: 59926,
        EventType.POWER_WATER_OUTAGE: 26782,
    }
    
    def __init__(self, timeline: ClockTimeline, llm: LLMInterface, state_machine: StateMachine):
        """
        Initialize event factory
        
        Args:
            timeline: Timeline manager for scheduling events
            llm: LLM interface for generating descriptions
            state_machine: State machine for managing transitions
        """
        self.timeline = timeline
        self.llm = llm
        self.state_machine = state_machine
        self.event_counter = 0
        self.active_events: Dict[str, Event] = {}
    
    def generate_poisson_events(self, beta: float = 1.0):
        """
        不再生成泊松事件 → 直接导入并使用前61个模拟事件
        """ 
    # 获取完整模拟时间序列，并只取前60个
        sim_ts = get_simulated_ts()
        top60_event_times = sim_ts[:61]  # 前60个事件时间

    # 映射到时间线中
        if len(top60_event_times) > 1:
        # 原始时间最大、最小值
            t_min = top60_event_times.min()
            t_max = top60_event_times.max()
        
        # 线性映射公式：把 [t_min, t_max] → 映射到 [0, 60]
            top60_event_times = 60 * (top60_event_times - t_min) / (t_max - t_min)

    # 遍历前60个时间，创建事件
        for event_time in top60_event_times:
        # 确保时间不超过时间上限
            if event_time < self.timeline.time_limit:
            # 创建事件
                event = self._create_random_event(event_time)
                self.active_events[event.event_id] = event

            # 添加到时间线
                self.timeline.add_entry(
                    time=event_time,
                    entry_type="event_start",
                    data={
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "location": event.location
                    }
                )

            # 调度状态转换
                self._schedule_next_state_transition(event)
    
    def _create_random_event(self, start_time: float) -> Event:
        """
        Create a random event with LLM-generated description
        
        Args:
            start_time: When the event starts
            
        Returns:
            Newly created event
        """
        self.event_counter += 1
        event_id = f"EVENT_{self.event_counter:04d}"
        
        # Select event type according to empirical prior distribution.
        event_type = self._sample_event_type()
        
        # Generate random location based on event type
        location = self._generate_location(event_type)
        
        # Create event
        event = Event(
            event_id=event_id,
            event_type=event_type,
            start_time=start_time,
            location=location
        )
        
        # Generate initial situation description using LLM
        initial_situation_desc = self.llm.generate_situation_description(event_type, location)
        
        # Create initial SituationUpdate using StateMachine
        initial_situation = self.state_machine.create_situation_update(
            event, start_time, initial_situation_desc, EventState.INITIAL
        )
        event.situation_history.append(initial_situation)
        
        return event

    def _sample_event_type(self) -> EventType:
        """Sample event type from the empirical prior distribution."""
        event_types = list(EventType)
        counts = np.array(
            [self.EVENT_TYPE_PRIOR_COUNTS[event_type] for event_type in event_types],
            dtype=float
        )
        probabilities = counts / counts.sum()
        return np.random.choice(event_types, p=probabilities)
    
    def _generate_location(self, event_type: EventType) -> str:
        """
        Generate appropriate location based on event type
        
        Args:
            event_type: Type of emergency event
            
        Returns:
            Appropriate location string
        """
        location_mapping = {
            EventType.BUILDING_FIRE: [
                "市中心商业大厦", "住宅小区A栋", "办公楼", "学校教学楼", "医院住院部"
            ],
            EventType.FOREST_FIRE: [
                "北山森林公园", "西郊林地", "东山风景区", "南岭保护区", "城郊林场"
            ],
            EventType.VEHICLE_FIRE: [
                "环城高速公路", "市中心停车场", "加油站", "地下车库", "公交车站"
            ],
            EventType.GAS_LEAK: [
                "工业园区", "老旧住宅区", "餐饮街", "化工厂", "燃气站"
            ],
            EventType.HIGH_ALTITUDE_RESCUE: [
                "高层建筑工地", "电视塔", "高压线塔", "深井", "山崖"
            ],
            EventType.ELEVATOR_RESCUE: [
                "商场电梯", "办公楼电梯", "住宅楼电梯", "地铁站电梯", "医院电梯"
            ],
            EventType.FLOOD_DRAINAGE: [
                "低洼地区", "地下通道", "河边道路", "老城区", "排水系统"
            ],
            EventType.WATER_RESCUE: [
                "人工湖", "河流", "水库", "游泳池", "港口码头"
            ],
            EventType.WILDLIFE_HANDLING: [
                "住宅小区", "学校操场", "公园", "市场", "停车场"
            ],
            EventType.BURGLARY: [
                "住宅小区", "商店", "办公楼", "仓库", "别墅区"
            ],
            EventType.MISSING_CHILD: [
                "游乐场", "商场", "公园", "学校附近", "居民区"
            ],
            EventType.MISSING_PET: [
                "公园", "住宅小区", "街道", "宠物店附近", "绿化带"
            ],
            EventType.NOISE_DISTURBANCE: [
                "居民区", "商业街", "工地", "娱乐场所", "学校周边"
            ],
            EventType.POWER_WATER_OUTAGE: [
                "老旧住宅区", "工业区", "商业中心", "学校", "医院"
            ]
        }
        
        locations = location_mapping.get(event_type, [
            "市中心", "住宅区", "工业园", "学校", "医院"
        ])
        
        return np.random.choice(locations)
    
    def _schedule_next_state_transition(self, event: Event):
        """
        Schedule the next state transition for an event
        
        Args:
            event: Event to schedule transition for
        """
        if event.current_state != EventState.EXTINGUISHED:
            next_time = self.state_machine.get_next_state_time(
                event.next_state_time or event.start_time,
                event.event_type.value
            )
            event.next_state_time = next_time
            
            # Only schedule if within timeline limit
            if next_time <= self.timeline.time_limit:
                self.timeline.add_entry(
                    time=next_time,
                    entry_type="state_change",
                    data={"event_id": event.event_id}
                )
    
    def process_state_transition(self, event_id: str, transition_time: float):
        """
        Process a state transition for an event
        
        Args:
            event_id: ID of the event to transition
            transition_time: Time when transition occurs
        """
        if event_id not in self.active_events:
            return
        
        event = self.active_events[event_id]
        old_state = event.current_state
        new_state = self.state_machine.transition_state(old_state)
        
        # Generate new situation description if state changed
        if new_state != old_state:
            # Use LLM to generate more detailed state transition description
            try:
                situation_desc = self.llm.generate_state_update_description(event, new_state.value)
            except Exception:
                situation_desc = f"事件状态从{old_state.value}转变为{new_state.value}"
            
            # Update event state
            event.current_state = new_state
            
            # Create situation update using StateMachine
            situation_update = self.state_machine.create_situation_update(
                event, transition_time, situation_desc, new_state
            )
            event.situation_history.append(situation_update)
            
            # Schedule next transition if not extinguished
            if new_state != EventState.EXTINGUISHED:
                self._schedule_next_state_transition(event)
    
    def get_event_by_id(self, event_id: str) -> Event:
        """
        Get event by ID
        
        Args:
            event_id: Event identifier
            
        Returns:
            Event object or None if not found
        """
        return self.active_events.get(event_id)
    
    def get_active_events(self) -> Dict[str, Event]:
        """
        Get all active events
        
        Returns:
            Dictionary of active events
        """
        return self.active_events.copy()
    
    def get_events_by_type(self, event_type: EventType) -> list:
        """
        Get all events of a specific type
        
        Args:
            event_type: Type of events to retrieve
            
        Returns:
            List of events matching the type
        """
        return [
            event for event in self.active_events.values() 
            if event.event_type == event_type
        ]
    
    def get_events_by_state(self, state: EventState) -> list:
        """
        Get all events in a specific state
        
        Args:
            state: State to filter by
            
        Returns:
            List of events in the specified state
        """
        return [
            event for event in self.active_events.values() 
            if event.current_state == state
        ]
    
    def get_event_statistics(self) -> dict:
        """
        Get statistics about generated events
        
        Returns:
            Dictionary with event statistics
        """
        total_events = len(self.active_events)
        
        # Count by type
        type_counts = {}
        for event in self.active_events.values():
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        # Count by state
        state_counts = {}
        for event in self.active_events.values():
            state = event.current_state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            'total_events': total_events,
            'events_by_type': type_counts,
            'events_by_state': state_counts,
            'average_visibility': np.mean([e.situation_history[-1].visibility for e in self.active_events.values() if e.situation_history]) if total_events > 0 else 0,
            'average_impact': np.mean([e.situation_history[-1].impact for e in self.active_events.values() if e.situation_history]) if total_events > 0 else 0,
            'average_severity': np.mean([e.situation_history[-1].severity for e in self.active_events.values() if e.situation_history]) if total_events > 0 else 0
        }
