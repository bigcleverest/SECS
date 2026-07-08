"""
Timeline Management for Fire Alarm System
消防报警系统时间轴管理

Manages the simulation timeline from 0 to time_limit, maintaining chronological order
of all events and system state changes.
"""

from typing import List
from models import TimelineEntry


class ClockTimeline:
    """Manages simulation timeline from 0 to time_limit"""
    
    def __init__(self, time_limit: float):
        """
        Initialize timeline with specified time limit
        
        Args:
            time_limit: Maximum simulation time
        """
        self.time_limit = time_limit
        self.current_time = 0.0
        self.timeline: List[TimelineEntry] = []
    
    def add_entry(self, time: float, entry_type: str, data: dict):
        """
        Add entry to timeline maintaining chronological order
        
        Args:
            time: Time when the entry occurs
            entry_type: Type of entry (e.g., "event_start", "state_change", "alarm_call")
            data: Associated data for the entry
        """
        if time <= self.time_limit:
            entry = TimelineEntry(time=time, entry_type=entry_type, data=data)
            self.timeline.append(entry)
            self.timeline.sort(key=lambda x: x.time)
    
    def get_entries_up_to(self, time: float) -> List[TimelineEntry]:
        """
        Get all entries up to specified time
        
        Args:
            time: Cut-off time
            
        Returns:
            List of timeline entries up to the specified time
        """
        return [entry for entry in self.timeline if entry.time <= time]
    
    def get_entries_in_range(self, start_time: float, end_time: float) -> List[TimelineEntry]:
        """
        Get all entries within a time range
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of timeline entries within the specified range
        """
        return [
            entry for entry in self.timeline 
            if start_time <= entry.time <= end_time
        ]
    
    def get_entries_by_type(self, entry_type: str) -> List[TimelineEntry]:
        """
        Get all entries of a specific type
        
        Args:
            entry_type: Type of entry to filter by
            
        Returns:
            List of timeline entries matching the specified type
        """
        return [entry for entry in self.timeline if entry.entry_type == entry_type]
    
    def clear_timeline(self):
        """Clear all entries from the timeline"""
        self.timeline.clear()
        self.current_time = 0.0
    
    def get_timeline_summary(self) -> dict:
        """
        Get a summary of the timeline
        
        Returns:
            Dictionary containing timeline statistics
        """
        entry_types = {}
        for entry in self.timeline:
            entry_types[entry.entry_type] = entry_types.get(entry.entry_type, 0) + 1
        
        return {
            'total_entries': len(self.timeline),
            'time_limit': self.time_limit,
            'current_time': self.current_time,
            'entry_types': entry_types,
            'first_entry_time': self.timeline[0].time if self.timeline else None,
            'last_entry_time': self.timeline[-1].time if self.timeline else None
        }