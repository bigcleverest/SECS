"""
Caller Farm for Fire Alarm System
消防报警系统报警人工厂

Factory for generating alarm calls based on event characteristics.
Creates diverse caller profiles with varying uncertainty, emotion, and description styles.
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict
from models import Event, AlarmCall, CallerRole, SituationUpdate
from timeline import ClockTimeline
from llm_interface import LLMInterface


class CallerFarm:
    """Factory for generating alarm calls based on event characteristics"""
    
    def __init__(self, timeline: ClockTimeline, llm: LLMInterface):
        """
        Initialize caller farm
        
        Args:
            timeline: Timeline manager for scheduling calls
            llm: LLM interface for generating call descriptions
        """
        self.timeline = timeline
        self.llm = llm
        self.call_counter = 0
        self.active_callers: Dict[str, List[AlarmCall]] = defaultdict(list)
        
        # Configuration for caller generation
        self.caller_role_probabilities = {
            CallerRole.VICTIM: 0.3,        # 30% victims/trapped persons
            CallerRole.BYSTANDER: 0.5,     # 50% bystanders/observers
            CallerRole.REPRESENTATIVE: 0.2  # 20% representatives
        }
    
    def generate_alarm_calls(self, event: Event, duration: float):
        """
        Generate alarm calls for an event based on its characteristics
        
        Args:
            event: Event to generate calls for
            duration: Duration over which to generate calls
        """
        # Calculate call generation rate λ based on event characteristics
        lambda_rate = self._calculate_call_rate(event)
        
        current_time = event.start_time
        end_time = min(event.start_time + duration, self.timeline.time_limit)
        
        while current_time < end_time:
            # Generate next call time using exponential distribution
            if lambda_rate > 0:
                inter_call_time = np.random.exponential(1.0 / lambda_rate)
            else:
                break  # No calls if rate is 0
            
            current_time += inter_call_time
            
            if current_time < end_time:
                alarm_call = self._create_alarm_call(event, current_time)
                self.active_callers[event.event_id].append(alarm_call)
                
                # Add to timeline
                self.timeline.add_entry(
                    time=current_time,
                    entry_type="alarm_call",
                    data={
                        "call_id": alarm_call.call_id,
                        "event_id": alarm_call.event_id,
                        "caller_role": alarm_call.caller_role.value,
                        "uncertainty": alarm_call.uncertainty_level,
                        "emotional": alarm_call.emotional_level
                    }
                )
    
    def _calculate_call_rate(self, event: Event) -> float:
        """
        Calculate call generation rate based on event characteristics
        
        Args:
            event: Event to calculate rate for
            
        Returns:
            Call generation rate (calls per time unit)
        """
        base_rate = 0.1  # Base rate per time unit
        
        # Rate influenced by situation characteristics
        current_situation = None
        if event.situation_history:
            current_situation = event.situation_history[-1]
        
        if current_situation is not None:
            visibility_factor = current_situation.visibility * 2.0    # Higher visibility = more calls
            impact_factor = current_situation.impact * 1.5            # Higher impact = more calls
            severity_factor = current_situation.severity * 1.0        # Higher severity = more calls
        else:
            # Default factors if no situation available
            visibility_factor = impact_factor = severity_factor = 1.0
        
        # State-specific modifiers
        state_modifiers = {
            "初始态": 0.5,    # Lower call rate initially
            "冒烟": 1.0,      # Standard rate when smoke visible
            "明火": 1.8,      # High rate when flames visible
            "轰燃": 2.5,      # Very high rate during flashover
            "受控": 0.7,      # Reduced rate when controlled
            "扑灭": 0.2       # Very low rate when extinguished
        }
        
        state_modifier = state_modifiers.get(event.current_state.value, 1.0)
        
        lambda_rate = base_rate * (
            visibility_factor + impact_factor + severity_factor
        ) * state_modifier
        
        return max(0.0, lambda_rate)  # Ensure non-negative rate
    
    def _create_alarm_call(self, event: Event, call_time: float) -> AlarmCall:
        """
        Create an individual alarm call with LLM-generated description
        
        Args:
            event: Event being reported
            call_time: Time when call is made
            
        Returns:
            Generated alarm call
        """
        self.call_counter += 1
        call_id = f"CALL_{self.call_counter:04d}"
        
        # Get current situation for dynamic caller role selection
        current_situation = None
        if event.situation_history:
            current_situation = event.situation_history[-1]
        
        # Select caller role based on situation characteristics
        caller_role = self._select_caller_role(current_situation)
        
        # Generate caller characteristics based on role and situation
        uncertainty, emotional = self._generate_caller_characteristics(caller_role, event, current_situation)
        
        # Generate call description using LLM
        description = self.llm.generate_alarm_description(
            event, caller_role, uncertainty, emotional
        )
        
        return AlarmCall(
            call_id=call_id,
            event_id=event.event_id,
            caller_role=caller_role,
            call_time=call_time,
            description=description,
            uncertainty_level=uncertainty,
            emotional_level=emotional
        )
    
    def _select_caller_role(self, situation: SituationUpdate = None) -> CallerRole:
        """
        Select caller role based on situation characteristics or static probabilities
        
        Args:
            situation: Current situation update (if None, uses static probabilities)
        
        Returns:
            Selected caller role
        """
        if situation is not None:
            # Dynamic probability calculation based on situation characteristics
            # Higher impact/severity situations more likely to have victims
            # Higher visibility situations more likely to have bystanders
            victim_prob = situation.impact * 0.4 + situation.severity * 0.3
            bystander_prob = situation.visibility * 0.5
            representative_prob = 0.2  # Base probability for representatives
            
            # Normalize probabilities
            total = victim_prob + bystander_prob + representative_prob
            if total > 0:
                victim_prob /= total
                bystander_prob /= total
                representative_prob /= total
            else:
                # Fallback to equal probabilities
                victim_prob = bystander_prob = representative_prob = 1/3
            
            return np.random.choice(
                [CallerRole.VICTIM, CallerRole.BYSTANDER, CallerRole.REPRESENTATIVE],
                p=[victim_prob, bystander_prob, representative_prob]
            )
        else:
            # Fall back to static probabilities
            roles = list(self.caller_role_probabilities.keys())
            probabilities = list(self.caller_role_probabilities.values())
            return np.random.choice(roles, p=probabilities)
    
    def _generate_caller_characteristics(self, caller_role: CallerRole, event: Event, situation: SituationUpdate = None) -> tuple:
        """
        Generate uncertainty and emotional levels based on caller role and situation
        
        Args:
            caller_role: Role of the caller
            event: Event being reported
            situation: Current situation update (if None, uses default adjustments)
            
        Returns:
            Tuple of (uncertainty_level, emotional_level)
        """
        # Base characteristics by role
        if caller_role == CallerRole.VICTIM:
            # Victims: Low uncertainty (they're experiencing it), high emotion
            uncertainty_base = 0.2
            emotional_base = 0.8
        elif caller_role == CallerRole.BYSTANDER:
            # Bystanders: Medium uncertainty (observing from distance), medium emotion
            uncertainty_base = 0.5
            emotional_base = 0.5
        else:  # REPRESENTATIVE
            # Representatives: Lower uncertainty (collecting info), controlled emotion
            uncertainty_base = 0.3
            emotional_base = 0.4
        
        # Adjust based on situation characteristics or defaults
        if situation is not None:
            severity_adjustment = (situation.severity - 0.5) * 0.3  # ±0.15 adjustment
            visibility_adjustment = (situation.visibility - 0.5) * 0.2  # ±0.1 adjustment
            impact_adjustment = (situation.impact - 0.5) * 0.2  # ±0.1 adjustment
        else:
            # Use default values if no situation available
            severity_adjustment = visibility_adjustment = impact_adjustment = 0.0
        
        # Uncertainty decreases with visibility, increases with severity if unclear
        uncertainty = uncertainty_base - visibility_adjustment + max(0, severity_adjustment * 0.5)
        
        # Emotion increases with severity, visibility, and impact
        emotional = emotional_base + severity_adjustment + visibility_adjustment * 0.5 + impact_adjustment * 0.3
        
        # Add random variation and clamp to [0, 1]
        uncertainty = max(0.0, min(1.0, uncertainty + np.random.normal(0, 0.1)))
        emotional = max(0.0, min(1.0, emotional + np.random.normal(0, 0.1)))
        
        return uncertainty, emotional
    
    def update_call_rates(self, event: Event, new_duration: float):
        """
        Update call generation rates when event state changes
        
        Args:
            event: Event with updated characteristics
            new_duration: Duration for which to generate additional calls
        """
        # Generate additional calls for the updated event state
        self.generate_alarm_calls(event, new_duration)
    
    def get_calls_for_event(self, event_id: str) -> List[AlarmCall]:
        """
        Get all calls for a specific event
        
        Args:
            event_id: Event identifier
            
        Returns:
            List of alarm calls for the event
        """
        return self.active_callers.get(event_id, [])
    
    def get_all_calls(self) -> List[AlarmCall]:
        """
        Get all alarm calls across all events
        
        Returns:
            List of all alarm calls
        """
        all_calls = []
        for calls in self.active_callers.values():
            all_calls.extend(calls)
        return sorted(all_calls, key=lambda x: x.call_time)
    
    def get_calls_by_role(self, role: CallerRole) -> List[AlarmCall]:
        """
        Get all calls made by a specific caller role
        
        Args:
            role: Caller role to filter by
            
        Returns:
            List of calls made by the specified role
        """
        all_calls = self.get_all_calls()
        return [call for call in all_calls if call.caller_role == role]
    
    def get_calls_in_time_range(self, start_time: float, end_time: float) -> List[AlarmCall]:
        """
        Get all calls within a specific time range
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of calls within the time range
        """
        all_calls = self.get_all_calls()
        return [
            call for call in all_calls 
            if start_time <= call.call_time <= end_time
        ]
    
    def get_call_statistics(self) -> dict:
        """
        Get statistics about generated calls
        
        Returns:
            Dictionary with call statistics
        """
        all_calls = self.get_all_calls()
        total_calls = len(all_calls)
        
        if total_calls == 0:
            return {
                'total_calls': 0,
                'calls_by_role': {},
                'calls_by_event': {},
                'average_uncertainty': 0,
                'average_emotional': 0,
                'call_rate_by_event': {}
            }
        
        # Count by role
        role_counts = {}
        for call in all_calls:
            role = call.caller_role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Count by event
        event_counts = {}
        for event_id, calls in self.active_callers.items():
            event_counts[event_id] = len(calls)
        
        # Calculate averages
        avg_uncertainty = np.mean([call.uncertainty_level for call in all_calls])
        avg_emotional = np.mean([call.emotional_level for call in all_calls])
        
        # Calculate call rates per event
        call_rates = {}
        for event_id, calls in self.active_callers.items():
            if calls:
                duration = max(call.call_time for call in calls) - min(call.call_time for call in calls)
                if duration > 0:
                    call_rates[event_id] = len(calls) / duration
                else:
                    call_rates[event_id] = len(calls)  # All calls at same time
        
        return {
            'total_calls': total_calls,
            'calls_by_role': role_counts,
            'calls_by_event': event_counts,
            'average_uncertainty': float(avg_uncertainty),
            'average_emotional': float(avg_emotional),
            'call_rate_by_event': call_rates
        }
    
    def generate_call_summary(self, event_id: str) -> str:
        """
        Generate a summary of calls for a specific event
        
        Args:
            event_id: Event identifier
            
        Returns:
            Human-readable summary string
        """
        calls = self.get_calls_for_event(event_id)
        
        if not calls:
            return f"事件 {event_id}: 无报警电话"
        
        role_counts = {}
        for call in calls:
            role = call.caller_role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        
        avg_uncertainty = np.mean([call.uncertainty_level for call in calls])
        avg_emotional = np.mean([call.emotional_level for call in calls])
        
        summary = f"事件 {event_id}: 共{len(calls)}个报警电话\n"
        summary += f"  角色分布: {role_counts}\n"
        summary += f"  平均不确定度: {avg_uncertainty:.2f}\n"
        summary += f"  平均情绪度: {avg_emotional:.2f}"
        
        return summary