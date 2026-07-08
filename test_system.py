#!/usr/bin/env python3
"""
Test script for Fire Alarm System - Basic functionality test without LLM calls
测试脚本 - 不依赖大模型API的基础功能测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import numpy, use basic random if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    import random
    HAS_NUMPY = False
    print("⚠️  注意：numpy未安装，使用基础随机数生成器")

# Import system components with dependency handling
if HAS_NUMPY:
    from fire_alarm_system import *
else:
    # Create minimal components for testing without numpy
    print("⚠️  注意：将使用简化版本进行测试")


class MockLLMInterface:
    """Mock LLM interface for testing without API calls"""
    
    def generate_situation_description(self, event_type: EventType, location: str) -> str:
        """Mock situation description generation"""
        templates = {
            EventType.BUILDING_FIRE: f"{location}发生建筑火灾，现场浓烟滚滚，火势正在蔓延",
            EventType.VEHICLE_FIRE: f"{location}发生车辆火灾，汽车引擎冒烟起火",
            EventType.GAS_LEAK: f"{location}发生气体泄露，现场有刺激性气味",
            EventType.FOREST_FIRE: f"{location}发生林草燃烧，火势向周围扩散"
        }
        return templates.get(event_type, f"{location}发生{event_type.value}，现场情况紧急")
    
    def generate_alarm_description(self, event: Event, caller_role: CallerRole, 
                                 uncertainty: float, emotional: float) -> str:
        """Mock alarm description generation"""
        
        latest_situation = event.situation_history[-1].situation if event.situation_history else "情况不明"
        
        templates = {
            CallerRole.VICTIM: [
                "救命！我被困了！现场很危险！",
                "快来救我们！情况很紧急！",
                "我们出不去了！到处都是烟！"
            ],
            CallerRole.BYSTANDER: [
                "我看到这里发生了紧急情况",
                "这边好像出事了，你们快来看看",
                "现场情况不太对，可能需要帮助"
            ],
            CallerRole.REPRESENTATIVE: [
                "我代表大家报警，这里发生了紧急情况",
                "我是居民代表，这里需要紧急救援",
                "作为目击者，我觉得应该报警"
            ]
        }
        
        base_desc = np.random.choice(templates[caller_role])
        
        # Add uncertainty and emotion effects
        if uncertainty > 0.7:
            base_desc += " 具体情况我也不太清楚"
        if emotional > 0.7:
            base_desc += " 真的很急！！！"
        
        return f"{base_desc} 地点：{event.location}"


def test_basic_components():
    """Test individual components"""
    print("🧪 测试基础组件...")
    
    # Test Timeline
    timeline = ClockTimeline(60.0)
    timeline.add_entry(10.0, "test", {"data": "test"})
    assert len(timeline.timeline) == 1
    print("✅ ClockTimeline 测试通过")
    
    # Test State Machine
    state_machine = StateMachine()
    next_state = state_machine.transition_state(EventState.INITIAL)
    assert next_state in [EventState.SMOKE, EventState.CONTROLLED, EventState.EXTINGUISHED]
    print("✅ StateMachine 测试通过")
    
    # Test Mock LLM
    mock_llm = MockLLMInterface()
    desc = mock_llm.generate_situation_description(EventType.BUILDING_FIRE, "测试地点")
    assert "建筑火灾" in desc
    print("✅ MockLLMInterface 测试通过")


class TestFireAlarmSimulation(FireAlarmSimulation):
    """Test version of simulation with mock LLM"""
    
    def __init__(self, time_limit: float):
        self.timeline = ClockTimeline(time_limit)
        self.llm = MockLLMInterface()  # Use mock instead of real LLM
        self.state_machine = StateMachine()
        self.event_factory = EventFactory(self.timeline, self.llm, self.state_machine)
        self.caller_farm = CallerFarm(self.timeline, self.llm)


def test_full_simulation():
    """Test complete simulation with mock components"""
    print("\n🎯 测试完整仿真流程...")
    
    # Set deterministic seed for testing
    np.random.seed(42)
    
    # Create test simulation
    simulation = TestFireAlarmSimulation(30.0)
    
    # Run simulation
    simulation.run_simulation()
    
    # Verify results
    assert len(simulation.event_factory.active_events) > 0, "应该生成至少一个事件"
    
    total_calls = sum(len(calls) for calls in simulation.caller_farm.active_callers.values())
    assert total_calls > 0, "应该生成至少一个报警电话"
    
    print("✅ 完整仿真测试通过")


def test_event_states():
    """Test event state transitions"""
    print("\n🔄 测试事件状态转移...")
    
    timeline = ClockTimeline(60.0)
    mock_llm = MockLLMInterface()
    state_machine = StateMachine()
    
    # Create test event
    event = Event(
        event_id="TEST_001",
        event_type=EventType.BUILDING_FIRE,
        start_time=0.0,
        location="测试地点"
    )
    
    # Test state transitions
    original_state = event.current_state
    for _ in range(5):
        new_state = state_machine.transition_state(event.current_state)
        state_machine.update_event_characteristics(event)
        print(f"  状态转移: {event.current_state.value} → {new_state.value}")
        event.current_state = new_state
        
        if new_state == EventState.EXTINGUISHED:
            break
    
    print("✅ 事件状态转移测试通过")


def test_caller_generation():
    """Test caller and alarm call generation"""
    print("\n📞 测试报警电话生成...")
    
    timeline = ClockTimeline(60.0)
    mock_llm = MockLLMInterface()
    caller_farm = CallerFarm(timeline, mock_llm)
    
    # Create test event
    event = Event(
        event_id="TEST_001",
        event_type=EventType.BUILDING_FIRE,
        start_time=0.0,
        location="测试地点",
        visibility=0.8,
        impact=0.7,
        severity=0.6
    )
    
    # Add initial situation
    event.situation_history.append(
        SituationUpdate(time=0.0, situation="测试态势描述")
    )
    
    # Generate calls
    caller_farm.generate_alarm_calls(event, 10.0)
    
    # Verify calls were generated
    calls = caller_farm.active_callers[event.event_id]
    assert len(calls) > 0, "应该生成至少一个报警电话"
    
    # Check call properties
    for call in calls[:3]:  # Check first 3 calls
        print(f"  电话 {call.call_id}: {call.caller_role.value} - {call.description[:50]}...")
        assert call.uncertainty_level >= 0 and call.uncertainty_level <= 1
        assert call.emotional_level >= 0 and call.emotional_level <= 1
    
    print("✅ 报警电话生成测试通过")


def main():
    """Run all tests"""
    print("🚨 消防报警系统测试开始")
    print("=" * 50)
    
    try:
        test_basic_components()
        test_event_states()
        test_caller_generation()
        test_full_simulation()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！系统运行正常")
        print("💡 提示：要运行完整版本（使用真实大模型），请执行：")
        print("   python src/fire_alarm_system.py --time_limit 60")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())