# models/openai_client.py
import openai
import os
import asyncio
from typing import Dict, Any, Optional
import json

class OpenAIClient:
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = "gpt-4-1106-preview"  # Latest GPT-4 Turbo
        self.max_tokens = 4000
        self.temperature = 0.7
    
    async def generate_content(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate content using OpenAI GPT-4"""
        try:
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert personal statement consultant with years of experience helping students craft compelling narratives for graduate school applications. You understand how to balance authenticity with persuasive storytelling."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # 如果有上下文，添加到系统消息中
            if context:
                major = context.get("major", "")
                if major:
                    messages[0]["content"] += f" The student is applying for {major} programs."
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            return {
                "content": content,
                "model": self.model,
                "tokens_used": usage.total_tokens,
                "cost": self._calculate_cost(usage.total_tokens),
                "success": True
            }
            
        except Exception as e:
            return {
                "content": f"OpenAI API error: {str(e)}",
                "model": self.model,
                "tokens_used": 0,
                "cost": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on GPT-4 Turbo pricing"""
        # GPT-4 Turbo pricing: $0.01 per 1K input tokens, $0.03 per 1K output tokens
        # Approximate 50/50 split for estimation
        return (tokens / 1000) * 0.02

# models/claude_client.py
import anthropic
import os
from typing import Dict, Any

class ClaudeClient:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = "claude-3-sonnet-20240229"
        self.max_tokens = 4000
        self.temperature = 0.7
    
    async def generate_content(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate content using Claude"""
        try:
            # 构建系统提示
            system_prompt = "You are an expert academic writing consultant specializing in personal statements for graduate applications. You have a deep understanding of what admissions committees look for and how to craft authentic, compelling narratives that showcase a student's unique journey and potential."
            
            if context and context.get("major"):
                system_prompt += f" You are particularly skilled at helping students applying to {context['major']} programs."
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            content = response.content[0].text
            usage = response.usage
            
            return {
                "content": content,
                "model": self.model,
                "tokens_used": usage.input_tokens + usage.output_tokens,
                "cost": self._calculate_cost(usage.input_tokens, usage.output_tokens),
                "success": True
            }
            
        except Exception as e:
            return {
                "content": f"Claude API error: {str(e)}",
                "model": self.model,
                "tokens_used": 0,
                "cost": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on Claude pricing"""
        # Claude 3 Sonnet pricing: $3 per 1M input tokens, $15 per 1M output tokens
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost

# models/deepseek_client.py
import httpx
import os
from typing import Dict, Any
import json

class DeepSeekClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
        self.max_tokens = 4000
        self.temperature = 0.7
    
    async def generate_content(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate content using DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in academic writing and personal statement creation. You provide clear, structured, and actionable guidance."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed: {response.status_code}")
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                
                return {
                    "content": content,
                    "model": self.model,
                    "tokens_used": total_tokens,
                    "cost": self._calculate_cost(total_tokens),
                    "success": True
                }
                
        except Exception as e:
            return {
                "content": f"DeepSeek API error: {str(e)}",
                "model": self.model,
                "tokens_used": 0,
                "cost": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on DeepSeek pricing"""
        # DeepSeek pricing is very competitive, approximately $0.001 per 1K tokens
        return (tokens / 1000) * 0.001

# models/model_factory.py
from enum import Enum
from typing import Dict, Any, Union
from .openai_client import OpenAIClient
from .claude_client import ClaudeClient
from .deepseek_client import DeepSeekClient

class ModelType(str, Enum):
    OPENAI_GPT4 = "openai-gpt4"
    CLAUDE_SONNET = "claude-sonnet"
    DEEPSEEK = "deepseek"

class ModelFactory:
    def __init__(self):
        self._clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all AI model clients"""
        try:
            self._clients[ModelType.OPENAI_GPT4] = OpenAIClient()
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
        
        try:
            self._clients[ModelType.CLAUDE_SONNET] = ClaudeClient()
        except Exception as e:
            print(f"Failed to initialize Claude client: {e}")
        
        try:
            self._clients[ModelType.DEEPSEEK] = DeepSeekClient()
        except Exception as e:
            print(f"Failed to initialize DeepSeek client: {e}")
    
    async def generate_content(self, model_type: ModelType, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate content using the specified model"""
        if model_type not in self._clients:
            return {
                "content": f"Model {model_type} not available",
                "model": model_type,
                "tokens_used": 0,
                "cost": 0.0,
                "success": False,
                "error": f"Model {model_type} not configured"
            }
        
        client = self._clients[model_type]
        return await client.generate_content(prompt, context)
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return list(self._clients.keys())
    
    def is_model_available(self, model_type: ModelType) -> bool:
        """Check if a specific model is available"""
        return model_type in self._clients

# services/enhanced_model_selector.py
import os
from typing import Dict, Any, List
from models.model_factory import ModelType
import json

class EnhancedModelSelector:
    def __init__(self):
        self.model_capabilities = {
            ModelType.OPENAI_GPT4: {
                "strengths": ["creative_writing", "storytelling", "general_knowledge"],
                "cost_per_1k_tokens": 0.02,
                "speed": "medium",
                "quality": "high",
                "best_for": ["step7", "creative_tasks"]
            },
            ModelType.CLAUDE_SONNET: {
                "strengths": ["analytical", "structured_thinking", "academic_writing"],
                "cost_per_1k_tokens": 0.015,
                "speed": "medium", 
                "quality": "high",
                "best_for": ["step5", "step8", "analytical_tasks"]
            },
            ModelType.DEEPSEEK: {
                "strengths": ["cost_effective", "fast_response", "general_purpose"],
                "cost_per_1k_tokens": 0.001,
                "speed": "fast",
                "quality": "good",
                "best_for": ["step1", "step2", "step3", "step4", "step6"]
            }
        }
        
        # 专业领域偏好
        self.field_preferences = {
            "computer_science": {
                "step5": ModelType.CLAUDE_SONNET,  # 结构化大纲
                "step7": ModelType.OPENAI_GPT4,   # 创意文书
            },
            "business": {
                "step5": ModelType.OPENAI_GPT4,   # 商业思维
                "step7": ModelType.OPENAI_GPT4,   # 说服力文书
            },
            "engineering": {
                "step5": ModelType.CLAUDE_SONNET, # 技术分析
                "step7": ModelType.CLAUDE_SONNET, # 精确表达
            },
            "liberal_arts": {
                "step5": ModelType.OPENAI_GPT4,   # 创意大纲
                "step7": ModelType.OPENAI_GPT4,   # 文学性表达
            }
        }
    
    def select_model(self, step: str, context: Dict[str, Any], preferences: Dict[str, Any] = None) -> ModelType:
        """智能选择最适合的模型"""
        
        # 基础步骤偏好
        base_preferences = {
            "step1": ModelType.DEEPSEEK,
            "step2": ModelType.DEEPSEEK,
            "step3": ModelType.DEEPSEEK,
            "step4": ModelType.DEEPSEEK,
            "step5": ModelType.CLAUDE_SONNET,  # 大纲生成需要结构化思维
            "step6": ModelType.DEEPSEEK,
            "step7": ModelType.OPENAI_GPT4,    # 文书生成需要创意
            "step8": ModelType.CLAUDE_SONNET,  # 精细修改需要分析能力
        }
        
        # 考虑专业领域
        major = context.get("major", "").lower()
        field_type = self._categorize_field(major)
        
        if field_type in self.field_preferences:
            field_pref = self.field_preferences[field_type].get(step)
            if field_pref:
                return field_pref
        
        # 考虑用户预算偏好
        if preferences and preferences.get("cost_conscious", False):
            if step not in ["step7"]:  # 除了关键的文书生成步骤
                return ModelType.DEEPSEEK
        
        # 考虑质量偏好
        if preferences and preferences.get("quality_priority", False):
            if step in ["step5", "step7", "step8"]:
                return ModelType.CLAUDE_SONNET if step != "step7" else ModelType.OPENAI_GPT4
        
        return base_preferences.get(step, ModelType.DEEPSEEK)
    
    def _categorize_field(self, major: str) -> str:
        """根据专业名称分类"""
        major = major.lower()
        
        if any(term in major for term in ["computer", "计算机", "software", "cs", "tech", "engineering", "工程"]):
            return "computer_science"
        elif any(term in major for term in ["business", "商业", "management", "mba", "finance", "经济"]):
            return "business"
        elif any(term in major for term in ["literature", "文学", "art", "艺术", "history", "历史", "philosophy", "哲学"]):
            return "liberal_arts"
        else:
            return "general"
    
    def get_model_reasoning(self, step: str, selected_model: ModelType, context: Dict[str, Any]) -> str:
        """解释模型选择的原因"""
        major = context.get("major", "Unknown")
        capabilities = self.model_capabilities.get(selected_model, {})
        
        reasoning = f"为{step}选择了{selected_model.value}，因为它在{capabilities.get('strengths', [])}方面表现出色"
        
        if major != "Unknown":
            reasoning += f"，特别适合{major}专业的申请材料"
        
        return reasoning

# utils/cost_tracker.py
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

class CostTracker:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.daily_budget = float(os.getenv("DAILY_API_BUDGET", 50.0))
    
    async def track_usage(self, session_id: str, model: str, tokens: int, cost: float):
        """记录API使用情况"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 记录到会话
        session_key = f"session:{session_id}"
        session_data = json.loads(self.redis.get(session_key) or "{}")
        
        if "cost_tracking" not in session_data:
            session_data["cost_tracking"] = {"total_tokens": 0, "estimated_cost": 0.0, "calls": []}
        
        session_data["cost_tracking"]["total_tokens"] += tokens
        session_data["cost_tracking"]["estimated_cost"] += cost
        session_data["cost_tracking"]["calls"].append({
            "model": model,
            "tokens": tokens,
            "cost": cost,
            "timestamp": datetime.now().isoformat()
        })
        
        self.redis.setex(session_key, 86400, json.dumps(session_data))
        
        # 记录到日统计
        daily_key = f"usage:daily:{today}"
        daily_data = json.loads(self.redis.get(daily_key) or '{"total_cost": 0.0, "total_tokens": 0, "calls": 0}')
        daily_data["total_cost"] += cost
        daily_data["total_tokens"] += tokens
        daily_data["calls"] += 1
        
        self.redis.setex(daily_key, 86400 * 7, json.dumps(daily_data))  # 保存7天
    
    async def check_budget_limit(self) -> Dict[str, Any]:
        """检查是否超出预算限制"""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_key = f"usage:daily:{today}"
        daily_data = json.loads(self.redis.get(daily_key) or '{"total_cost": 0.0}')
        
        current_cost = daily_data.get("total_cost", 0.0)
        remaining_budget = self.daily_budget - current_cost
        
        return {
            "daily_budget": self.daily_budget,
            "used_today": current_cost,
            "remaining": remaining_budget,
            "exceeded": current_cost > self.daily_budget,
            "usage_percentage": (current_cost / self.daily_budget) * 100
        }