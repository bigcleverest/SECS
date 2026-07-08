"""
Main Simulation Engine for Fire Alarm System
消防报警系统主仿真引擎

Orchestrates the complete simulation including event generation, state transitions,
and alarm call generation with comprehensive reporting.
"""

from collections import defaultdict
from timeline import ClockTimeline
from llm_interface import LLMInterface
from state_machine import StateMachine
from event_factory import EventFactory
from caller_farm import CallerFarm
from models import Event, AlarmCall, CallerRole, EventState, TimelineEntry


class FireAlarmSimulation:
    """Main simulation orchestrator"""
    
    def __init__(self, time_limit: float, llm_interface: LLMInterface = None):
        """
        Initialize simulation with specified time limit
        
        Args:
            time_limit: Maximum simulation time
            llm_interface: LLM interface to use (defaults to new instance)
        """
        self.timeline = ClockTimeline(time_limit)
        self.llm = llm_interface or LLMInterface()
        self.state_machine = StateMachine()
        self.event_factory = EventFactory(self.timeline, self.llm, self.state_machine)
        self.caller_farm = CallerFarm(self.timeline, self.llm)
        
        # Simulation parameters
        self.event_generation_rate = 1.0  # β parameter for Poisson process
        self.initial_call_duration = 10.0  # Duration for initial call generation
        self.state_change_call_duration = 10.0  # Duration for calls after state changes
    
    def run_simulation(self, verbose: bool = True) -> dict:
        """
        Execute the complete simulation
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing simulation results and statistics
        """
        if verbose:
            print(f"🚨 启动消防报警系统仿真 (时限: {self.timeline.time_limit})")
            print("=" * 60)
        
        # Step 1: Generate events using Poisson process
        if verbose:
            print("🎲 生成随机事件...")
        self.event_factory.generate_poisson_events(beta=self.event_generation_rate)
        
        # Step 2: Generate initial alarm calls for each event
        if verbose:
            print("📞 生成初始报警电话...")
        for event in self.event_factory.active_events.values():
            # Generate calls for initial period (before first state change)
            initial_duration = min(
                self.initial_call_duration, 
                self.timeline.time_limit - event.start_time
            )
            self.caller_farm.generate_alarm_calls(event, initial_duration)
        
        # Step 3: Process timeline chronologically
        if verbose:
            print("⏱️  执行时间线仿真...")
        self._process_timeline()
        
        # Step 4: Generate final report
        results = self._generate_results()
        
        if verbose:
            self._print_report(results)
        
        return results
    
    def _process_timeline(self):
        """Process all timeline entries in chronological order"""
        for entry in self.timeline.timeline:
            if entry.entry_type == "state_change":
                event_id = entry.data["event_id"]
                self.event_factory.process_state_transition(event_id, entry.time)
                
                # Generate additional calls for the new state
                if event_id in self.event_factory.active_events:
                    event = self.event_factory.active_events[event_id]
                    remaining_time = self.timeline.time_limit - entry.time
                    if remaining_time > 0:
                        call_duration = min(self.state_change_call_duration, remaining_time)
                        self.caller_farm.update_call_rates(event, call_duration)
    
    def _generate_results(self) -> dict:
        """
        Generate comprehensive simulation results
        
        Returns:
            Dictionary containing all simulation statistics and data
        """
        # Get statistics from all components
        event_stats = self.event_factory.get_event_statistics()
        call_stats = self.caller_farm.get_call_statistics()
        timeline_stats = self.timeline.get_timeline_summary()
        
        # Calculate additional metrics
        total_events = len(self.event_factory.active_events)
        total_calls = call_stats['total_calls']
        calls_per_event = total_calls / total_events if total_events > 0 else 0
        
        # Get sample data for reporting
        sample_events = list(self.event_factory.active_events.values())[:5]
        sample_calls = self.caller_farm.get_all_calls()[:10]
        
        return {
            'simulation_parameters': {
                'time_limit': self.timeline.time_limit,
                'event_generation_rate': self.event_generation_rate,
                'initial_call_duration': self.initial_call_duration,
                'state_change_call_duration': self.state_change_call_duration
            },
            'summary_statistics': {
                'total_events': total_events,
                'total_calls': total_calls,
                'calls_per_event': calls_per_event,
                'simulation_duration': self.timeline.time_limit
            },
            'event_statistics': event_stats,
            'call_statistics': call_stats,
            'timeline_statistics': timeline_stats,
            'sample_events': [self._event_to_dict(event) for event in sample_events],
            'sample_calls': [self._call_to_dict(call) for call in sample_calls],
            'raw_data': {
                'events': {eid: self._event_to_dict(event) 
                          for eid, event in self.event_factory.active_events.items()},
                'calls': [self._call_to_dict(call) for call in self.caller_farm.get_all_calls()],
                'timeline': [self._timeline_entry_to_dict(entry) for entry in self.timeline.timeline]
            }
        }
    
    def _get_current_situation_summary(self, event: Event) -> dict:
        """Get summary of current situation"""
        if event.situation_history:
            current = event.situation_history[-1]
            return {
                'time': current.time,
                'situation': current.situation,
                'visibility': current.visibility,
                'impact': current.impact,
                'severity': current.severity
            }
        return {
            'time': event.start_time,
            'situation': 'Initial state',
            'visibility': 0.3,
            'impact': 0.2,
            'severity': 0.3
        }
    
    def _event_to_dict(self, event) -> dict:
        """Convert event object to dictionary"""
        return {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'start_time': event.start_time,
            'location': event.location,
            'current_state': event.current_state.value,
            'current_situation': self._get_current_situation_summary(event),
            'situation_history': [
                {
                    'time': su.time, 
                    'situation': su.situation,
                    'visibility': su.visibility,
                    'impact': su.impact,
                    'severity': su.severity
                } 
                for su in event.situation_history
            ]
        }
    
    def _call_to_dict(self, call) -> dict:
        """Convert call object to dictionary"""
        return {
            'call_id': call.call_id,
            'event_id': call.event_id,
            'caller_role': call.caller_role.value,
            'call_time': call.call_time,
            'description': call.description,
            'uncertainty_level': call.uncertainty_level,
            'emotional_level': call.emotional_level
        }
    
    def _timeline_entry_to_dict(self, entry) -> dict:
        """Convert timeline entry to dictionary"""
        return {
            'time': entry.time,
            'entry_type': entry.entry_type,
            'data': entry.data
        }
    
    def _print_report(self, results: dict):
        """Print detailed simulation report"""
        print("\n" + "=" * 60)
        print("📊 仿真结果报告")
        print("=" * 60)
        
        # Summary statistics
        summary = results['summary_statistics']
        print(f"总事件数: {summary['total_events']}")
        print(f"总报警电话数: {summary['total_calls']}")
        print(f"平均每事件报警数: {summary['calls_per_event']:.1f}")
        
        # Event type distribution
        event_stats = results['event_statistics']
        if 'events_by_type' in event_stats:
            print(f"\n📈 事件类型分布:")
            for event_type, count in sorted(event_stats['events_by_type'].items()):
                print(f"  {event_type}: {count}")
        
        # Event state distribution
        if 'events_by_state' in event_stats:
            print(f"\n🔄 事件状态分布:")
            for state, count in sorted(event_stats['events_by_state'].items()):
                print(f"  {state}: {count}")
        
        # Call role distribution
        call_stats = results['call_statistics']
        if 'calls_by_role' in call_stats:
            print(f"\n👥 报警人角色分布:")
            for role, count in sorted(call_stats['calls_by_role'].items()):
                print(f"  {role}: {count}")
        
        # Sample events
        sample_events = results['sample_events']
        if sample_events:
            print(f"\n🔍 事件样例 (前{len(sample_events)}个):")
            for event in sample_events:
                print(f"\n事件 {event['event_id']}:")
                print(f"  类型: {event['event_type']}")
                print(f"  地点: {event['location']}")
                print(f"  开始时间: {event['start_time']:.1f}")
                print(f"  当前状态: {event['current_state']}")
                current_situation = event['current_situation']
                print(f"  可见性: {current_situation['visibility']:.2f}")
                print(f"  影响性: {current_situation['impact']:.2f}")
                print(f"  严重程度: {current_situation['severity']:.2f}")
                if event['situation_history']:
                    print(f"  初始态势: {event['situation_history'][0]['situation']}")
        
        # Sample calls
        sample_calls = results['sample_calls']
        if sample_calls:
            print(f"\n📞 报警电话样例 (前{len(sample_calls)}个):")
            for call in sample_calls:
                print(f"\n电话 {call['call_id']}:")
                print(f"  时间: {call['call_time']:.1f}")
                print(f"  角色: {call['caller_role']}")
                print(f"  描述: {call['description']}")
                print(f"  不确定度: {call['uncertainty_level']:.2f}")
                print(f"  情绪度: {call['emotional_level']:.2f}")
        
        # Performance metrics
        print(f"\n⚡ 性能指标:")
        print(f"  平均事件可见性: {event_stats.get('average_visibility', 0):.2f}")
        print(f"  平均事件影响性: {event_stats.get('average_impact', 0):.2f}")
        print(f"  平均事件严重程度: {event_stats.get('average_severity', 0):.2f}")
        print(f"  平均报警不确定度: {call_stats.get('average_uncertainty', 0):.2f}")
        print(f"  平均报警情绪度: {call_stats.get('average_emotional', 0):.2f}")
        
        # Timeline summary
        timeline_stats = results['timeline_statistics']
        print(f"\n⏰ 时间轴统计:")
        print(f"  总条目数: {timeline_stats['total_entries']}")
        if timeline_stats['entry_types']:
            for entry_type, count in timeline_stats['entry_types'].items():
                print(f"  {entry_type}: {count}")
    
    def export_results(self, filename: str, results: dict = None):
        """
        Export simulation results to JSON file
        
        Args:
            filename: Output filename
            results: Results to export (defaults to last simulation results)
        """
        import json
        
        if results is None:
            results = self.run_simulation(verbose=False)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📁 仿真结果已导出到: {filename}")
    
    def set_parameters(self, **kwargs):
        """
        Set simulation parameters
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'event_generation_rate' in kwargs:
            self.event_generation_rate = kwargs['event_generation_rate']
        if 'initial_call_duration' in kwargs:
            self.initial_call_duration = kwargs['initial_call_duration']
        if 'state_change_call_duration' in kwargs:
            self.state_change_call_duration = kwargs['state_change_call_duration']
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.timeline.clear_timeline()
        self.event_factory.active_events.clear()
        self.event_factory.event_counter = 0
        self.caller_farm.active_callers.clear()
        self.caller_farm.call_counter = 0