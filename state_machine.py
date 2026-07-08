"""
State Machine for Fire Alarm System
消防报警系统状态机

Manages event state transitions using Markov process with different states having
varying visibility, impact, and severity characteristics.
"""

import numpy as np
from models import Event, EventState, SituationUpdate


class StateMachine:
    """Manages event state transitions using Markov process"""
    
    def __init__(self):
        """Initialize state machine with transition matrix and characteristics"""
        # State transition probabilities μ (simplified example)
        # In reality, this would be more complex and event-type specific
        self.transition_matrix = {
            EventState.INITIAL: {
                EventState.SMOKE: 0.7,
                EventState.CONTROLLED: 0.2,
                EventState.EXTINGUISHED: 0.1
            },
            EventState.SMOKE: {
                EventState.OPEN_FLAME: 0.6,
                EventState.CONTROLLED: 0.3,
                EventState.EXTINGUISHED: 0.1
            },
            EventState.OPEN_FLAME: {
                EventState.FLASHOVER: 0.4,
                EventState.CONTROLLED: 0.4,
                EventState.EXTINGUISHED: 0.2
            },
            EventState.FLASHOVER: {
                EventState.CONTROLLED: 0.6,
                EventState.EXTINGUISHED: 0.4
            },
            EventState.CONTROLLED: {
                EventState.EXTINGUISHED: 0.8,
                EventState.CONTROLLED: 0.2  # Stay controlled
            },
            EventState.EXTINGUISHED: {
                EventState.EXTINGUISHED: 1.0  # Terminal state
            }
        }
        
        # State characteristics affecting visibility, impact, severity
        self.state_characteristics = {
            EventState.INITIAL: {"visibility": 0.3, "impact": 0.2, "severity": 0.3},
            EventState.SMOKE: {"visibility": 0.7, "impact": 0.4, "severity": 0.5},
            EventState.OPEN_FLAME: {"visibility": 0.9, "impact": 0.7, "severity": 0.8},
            EventState.FLASHOVER: {"visibility": 1.0, "impact": 0.9, "severity": 1.0},
            EventState.CONTROLLED: {"visibility": 0.5, "impact": 0.3, "severity": 0.4},
            EventState.EXTINGUISHED: {"visibility": 0.1, "impact": 0.1, "severity": 0.1}
        }
    
    def get_next_state_time(self, current_time: float, event_type: str = None) -> float:
        """
        Calculate next state transition time using exponential distribution
        
        Args:
            current_time: Current simulation time
            event_type: Type of event (can influence transition timing)
            
        Returns:
            Time when next state transition should occur
        """
        # Mean time between state changes (can be adjusted per event type)
        # Different event types could have different transition rates
        base_interval = 10.0  # 10 time units on average
        
        # Event-specific adjustments could be added here
        if event_type:
            # Example: fires progress faster than other events
            if "火灾" in event_type:
                base_interval *= 0.8  # 20% faster progression
            elif "被困" in event_type:
                base_interval *= 1.5  # 50% slower progression
        
        return current_time + np.random.exponential(base_interval)
    
    def transition_state(self, current_state: EventState) -> EventState:
        """
        Perform Markov state transition
        
        Args:
            current_state: Current state of the event
            
        Returns:
            New state after transition
        """
        if current_state not in self.transition_matrix:
            return current_state
        
        transitions = self.transition_matrix[current_state]
        states = list(transitions.keys())
        probabilities = list(transitions.values())
        
        return np.random.choice(states, p=probabilities)
    
    def create_situation_update(self, event: Event, transition_time: float, 
                                situation_desc: str, new_state: EventState) -> SituationUpdate:
        """
        Create a new SituationUpdate with state characteristics
        
        Args:
            event: Event being updated
            transition_time: Time of the situation update
            situation_desc: Description of the new situation
            new_state: New state of the event
            
        Returns:
            New SituationUpdate object with appropriate characteristics
        """

        
        if new_state in self.state_characteristics:
            chars = self.state_characteristics[new_state]
            
            # Add some random variation while keeping within bounds
            visibility = max(0.0, min(1.0, chars["visibility"] + np.random.normal(0, 0.1)))
            impact = max(0.0, min(1.0, chars["impact"] + np.random.normal(0, 0.1)))
            severity = max(0.0, min(1.0, chars["severity"] + np.random.normal(0, 0.1)))
        else:
            # Default values if state not found
            visibility = impact = severity = 0.5
        
        return SituationUpdate(
            time=transition_time,
            situation=situation_desc,
            visibility=visibility,
            impact=impact,
            severity=severity
        )
    
    def get_state_progression_probability(self, from_state: EventState, to_state: EventState) -> float:
        """
        Get transition probability between two states
        
        Args:
            from_state: Starting state
            to_state: Target state
            
        Returns:
            Probability of transition (0.0 if not possible)
        """
        if from_state in self.transition_matrix:
            return self.transition_matrix[from_state].get(to_state, 0.0)
        return 0.0
    
    def get_terminal_states(self) -> list:
        """
        Get list of terminal states (states with no outgoing transitions)
        
        Returns:
            List of terminal states
        """
        terminal_states = []
        for state, transitions in self.transition_matrix.items():
            # Check if all transitions lead back to the same state
            if len(transitions) == 1 and state in transitions and transitions[state] == 1.0:
                terminal_states.append(state)
        return terminal_states
    
    def is_terminal_state(self, state: EventState) -> bool:
        """
        Check if a state is terminal
        
        Args:
            state: State to check
            
        Returns:
            True if state is terminal, False otherwise
        """
        return state in self.get_terminal_states()
    
    def get_current_situation(self, event: Event) -> SituationUpdate:
        """
        Get the current situation of an event (most recent SituationUpdate)
        
        Args:
            event: Event to get current situation for
            
        Returns:
            Most recent SituationUpdate or None if no history
        """
        if event.situation_history:
            return event.situation_history[-1]
        return None
    
    def get_state_description(self, state: EventState) -> str:
        """
        Get human-readable description of state characteristics
        
        Args:
            state: State to describe
            
        Returns:
            Description string
        """
        if state in self.state_characteristics:
            chars = self.state_characteristics[state]
            return f"{state.value}: 可见性={chars['visibility']:.1f}, 影响性={chars['impact']:.1f}, 严重程度={chars['severity']:.1f}"
        return f"{state.value}: 特征未定义"
    
    def simulate_state_sequence(self, initial_state: EventState, max_steps: int = 10) -> list:
        """
        Simulate a sequence of state transitions
        
        Args:
            initial_state: Starting state
            max_steps: Maximum number of transitions to simulate
            
        Returns:
            List of states in the transition sequence
        """
        sequence = [initial_state]
        current_state = initial_state
        
        for _ in range(max_steps):
            if self.is_terminal_state(current_state):
                break
            
            next_state = self.transition_state(current_state)
            sequence.append(next_state)
            current_state = next_state
        
        return sequence

 