"""
Large Language Model Interface for Fire Alarm System
消防报警系统大语言模型接口

Provides interface for LLM interactions to generate realistic situation descriptions
and alarm call content with appropriate variation and characteristics.
"""

import os

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI包未安装，将使用模拟模式")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from models import Event, EventType, CallerRole

DEFAULT_API_KEY = "sk-8666943ae667410fb46edb686184c37c"
DEFAULT_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3.7-plus")

class LLMInterface:
    """Interface for Large Language Model interactions"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize LLM interface with API configuration
        
        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the LLM service
        """

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI包未安装，请使用MockLLMInterface")
        
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY", DEFAULT_API_KEY)
        base_url = base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not api_key:
            raise ValueError("请提供 api_key，或在 DEFAULT_API_KEY 中填写 API Key。")

        self.legacy_openai = OpenAI is None
        if self.legacy_openai:
            openai.api_key = api_key
            openai.api_base = base_url
            self.client = openai
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _create_chat_completion(self, **kwargs):
        """Create a chat completion with either OpenAI SDK 0.x or 1.x."""
        if self.legacy_openai:
            return self.client.ChatCompletion.create(**kwargs)
        return self.client.chat.completions.create(**kwargs)

    def _get_message_content(self, response) -> str:
        if self.legacy_openai:
            return response["choices"][0]["message"]["content"].strip()
        return response.choices[0].message.content.strip()
    
    def generate_situation_description(self, event_type: EventType, location: str) -> str:
        """
        Generate initial situation description for an event
        
        Args:
            event_type: Type of emergency event
            location: Location where the event occurs
            
        Returns:
            Realistic situation description in Chinese
        """
        prompt = f"""
        您是一个紧急事件仿真器。请为以下紧急事件生成现实的初始情况描述：
        事件类型：{event_type.value}
        地点：{location}
        
        要求：
        - 用中文生成2-3句描述初始情况
        - 包含现场的具体细节
        - 保持逻辑一致性
        - 控制在150字符以内
        - 简洁但信息丰富
        """
        
        try:
            response = self._create_chat_completion(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an emergency situation generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return self._get_message_content(response)
        except Exception as e:
            # Fallback description
            return f"{location}发生{event_type.value}，现场情况不明，需要进一步确认。"
    
    def generate_alarm_description(self, event: Event, caller_role: CallerRole, 
                                 uncertainty: float, emotional: float) -> str:
        """
        Generate alarm call description based on caller characteristics
        
        Args:
            event: The event being reported
            caller_role: Role of the person making the call
            uncertainty: Level of uncertainty in the caller's report [0-1]
            emotional: Emotional level of the caller [0-1]
            
        Returns:
            Realistic alarm call description in Chinese
        """
        
        latest_situation = event.situation_history[-1].situation if event.situation_history else "情况不明"
        
        uncertainty_desc = "非常不确定" if uncertainty > 0.7 else "有些不确定" if uncertainty > 0.3 else "比较确定"
        emotional_desc = "非常紧张" if emotional > 0.7 else "有些紧张" if emotional > 0.3 else "相对冷静"
        
        prompt = f"""
        您正在模拟一个紧急报警电话。请根据报警人的角色特征生成现实的2-4句报警对话。应包含适当的情绪色彩、详细程度和不确定性。使用自然的语言模式，必要时可包含重复或犹豫。重点关注报警人从其位置能实际观察到的情况。
        
        事件：{event.event_type.value}
        地点：{event.location}
        当前情况：{latest_situation}
        报警人角色：{caller_role.value}
        不确定程度：{uncertainty_desc}
        情绪状态：{emotional_desc}
        
        要求：
        - 用中文生成
        - 反映报警人的角色视角
        - 适当包含不确定性和情绪因素
        - 根据不确定性调整描述长度（越不确定越简短模糊）
        - 包含一些可能遗漏或不清楚的细节
        - 控制在200字符以内
        - 听起来像真实的紧急报警电话
        """
        
        try:
            response = self._create_chat_completion(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are simulating an emergency caller."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8
            )
            return self._get_message_content(response)
        except Exception as e:
            # Fallback description
            role_prefix = {
                CallerRole.VICTIM: "我被困了！",
                CallerRole.BYSTANDER: "我看到有情况！",
                CallerRole.REPRESENTATIVE: "我代表大家报警，"
            }
            return f"{role_prefix.get(caller_role, '')} {event.location}发生{event.event_type.value}，请快来！"
    
    def generate_state_update_description(self, event: Event, new_state: str) -> str:
        """
        Generate description for event state changes
        
        Args:
            event: The event experiencing state change
            new_state: The new state description
            
        Returns:
            Description of the state change
        """
        prompt = f"""
        您是一个紧急情况仿真器。请生成简短的状态更新（2-3句），描述情况如何演变到新状态：
        
        事件：{event.event_type.value}，地点：{event.location}
        之前状态：{event.current_state.value}
        新状态：{new_state}
        
        要求：
        - 用中文生成
        - 控制在150字符以内
        - 具体说明发生了什么变化
        - 反映紧急情况的进展
        """
        
        try:
            response = self._create_chat_completion(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an emergency situation generator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.6
            )
            return self._get_message_content(response)
        except Exception as e:
            # Fallback description
            return f"{event.location}的{event.event_type.value}情况发生变化，进入{new_state}阶段"


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing without API calls"""
    
    def __init__(self):
        """Initialize mock interface without API client"""
        self.client = None
    
    def generate_situation_description(self, event_type: EventType, location: str) -> str:
        """Generate mock situation description"""
        templates = {
            EventType.BUILDING_FIRE: f"{location}发生建筑火灾，现场浓烟滚滚，火势正在蔓延",
            EventType.VEHICLE_FIRE: f"{location}发生车辆火灾，汽车引擎冒烟起火",
            EventType.GAS_LEAK: f"{location}发生气体泄露，现场有刺激性气味",
            EventType.FOREST_FIRE: f"{location}发生林草燃烧，火势向周围扩散",
            EventType.ELEVATOR_RESCUE: f"{location}电梯故障，有人员被困",
            EventType.WATER_RESCUE: f"{location}有人落水，情况紧急"
        }
        return templates.get(event_type, f"{location}发生{event_type.value}，现场情况紧急")
    
    def generate_alarm_description(self, event: Event, caller_role: CallerRole, 
                                 uncertainty: float, emotional: float) -> str:
        """Generate mock alarm description"""
        
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
        
        import random
        base_desc = random.choice(templates[caller_role])
        
        # Add uncertainty and emotion effects
        if uncertainty > 0.7:
            base_desc += " 具体情况我也不太清楚"
        if emotional > 0.7:
            base_desc += " 真的很急！！！"
        
        return f"{base_desc} 地点：{event.location}"
    
    def generate_state_update_description(self, event: Event, new_state: str) -> str:
        """Generate mock state update description"""
        return f"{event.location}的{event.event_type.value}情况发生变化，进入{new_state}阶段"
