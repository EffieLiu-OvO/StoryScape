from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import redis
import json
import uuid
import time
import asyncio
from datetime import datetime, timedelta
import os
from enum import Enum

# 导入AI模型库
import openai
import anthropic
import httpx

# Pydantic Models
class SessionRequest(BaseModel):
    user_id: Optional[str] = None
    major: Optional[str] = None
    school: Optional[str] = None

class ProcessRequest(BaseModel):
    session_id: str
    response: str
    step_override: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    current_step: str
    prompt: str
    generated_content: Optional[str] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None

class ModelType(str, Enum):
    OPENAI_GPT4 = "openai-gpt4"
    CLAUDE_SONNET = "claude-sonnet"
    DEEPSEEK = "deepseek"
    MOCK = "mock"

# FastAPI App
app = FastAPI(
    title="StoryScape API - Enhanced",
    description="AI-Powered Personal Statement Assistant with Natural Writing",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 环境配置
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Redis Configuration
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_client.ping()
    print("✅ Redis连接成功")
except Exception as e:
    print(f"⚠️ Redis连接失败: {e}")
    print("💡 将使用内存存储模式")
    redis_client = None

# Session Configuration
SESSION_EXPIRY = 24 * 60 * 60  # 24 hours
CACHE_EXPIRY = 7 * 24 * 60 * 60  # 7 days

# 内存存储（Redis备用方案）
memory_storage = {}

# 高质量文书创作的系统prompt模板
ENHANCED_PROMPTS = {
    "system_base": """你是一位经验丰富的文书顾问，擅长帮助学生撰写具有强烈个人特色和故事性的个人陈述。

你的写作风格特点：
1. 用具体的画面和细节而非抽象描述
2. 语言自然流畅，有情感温度，避免AI腔调
3. 善用比喻和生动的表达
4. 注重段落间的逻辑衔接和呼应
5. 每个故事都有具体的场景、冲突和收获

写作要求：
- 避免空洞的形容词堆砌
- 用数字、地点、具体事件支撑观点
- 保持真实感，像真人在讲述自己的经历
- 每段都要为整体故事服务
- 结尾要自然回扣开头""",

    "outline_generation": """基于学生的以下材料，为{major}专业的个人陈述创建一个5-6段的详细大纲。

学生背景：
专业：{major}
目标学校：{school}
背景材料：{background}
核心经历：{highlights}
动机目标：{motivation}

请创建一个有机统一的故事线，要求：

1. **找到一个贯穿全文的主题**：从学生经历中提炼出一个核心关注点或成长线索
2. **动机段要有画面感**：用具体的场景或转折点开头，而非抽象表述
3. **学术段要体现思维发展**：不只是课程罗列，要展现学生的认知演进
4. **实践段要有故事性**：每段都要有具体的挑战、解决过程和收获
5. **结尾要回扣主题**：自然地将个人成长与未来目标连接

请用markdown格式输出详细大纲，每段都要包含：
- 段落主题和作用
- 具体内容建议（包括可以使用的细节）
- 与前后段的衔接方式
- 避免的常见问题

大纲风格要像优秀的人类写作一样自然、有温度、有逻辑性。""",

    "essay_generation": """基于以下大纲和学生材料，撰写一篇完整的个人陈述。

确认的大纲：
{outline}

学生完整材料：
专业：{major}
学校：{school}
背景材料：{background}
核心经历：{highlights}
动机目标：{motivation}
大纲反馈：{outline_feedback}

写作要求：
1. **语言要自然有温度**：
   - 避免"我认为"、"我觉得"等AI常用表达
   - 用具体场景而非概念描述
   - 保持真实的情感流露

2. **内容要具体生动**：
   - 用准确的数字、地点、时间
   - 包含对话、感受、细节描写
   - 每个经历都要有具体的场景设置

3. **逻辑要流畅**：
   - 段与段之间要有自然过渡
   - 整体形成完整的成长故事
   - 结尾要呼应开头

4. **避免AI痕迹**：
   - 不用过于工整的排比句
   - 避免"深刻意识到"、"让我明白"等套话
   - 语言节奏要有变化，有长有短

请直接输出完整的个人陈述，字数控制在800-1000字。语言风格要像优秀的人类写作，不是AI生成的感觉。""",

    "refinement": """请根据以下反馈优化这份个人陈述：

原始文书：
{draft}

用户反馈：
{feedback}

优化要求：
1. 保持原有的故事结构和真实感
2. 根据反馈进行针对性修改
3. 确保修改后的语言更加自然流畅
4. 保持字数在合理范围内
5. 加强细节描写和画面感

请输出优化后的完整文书，重点改善用户提出的问题。"""
}

# AI模型客户端类
class OpenAIClient:
    def __init__(self):
        if OPENAI_API_KEY:
            self.client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
            self.model = "gpt-4-1106-preview"
            self.available = True
        else:
            self.client = None
            self.available = False
    
    async def generate_content(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.available:
            return {"content": "OpenAI API key not configured", "success": False, "error": "No API key"}
        
        try:
            # 构建更好的系统prompt
            system_prompt = """你是一位经验丰富的文书顾问，专门帮助学生撰写个人陈述。你的特长是：

1. 创作有故事性和画面感的文书，避免AI化的僵硬表达
2. 用具体的细节和场景来展现学生的经历，而不是抽象概括
3. 保持自然、有温度的语言风格
4. 构建逻辑清晰、结构完整的叙述

写作时请注意：
- 避免"我深刻意识到"、"让我明白"等AI常用套话
- 用具体的数字、地点、时间增强真实感
- 每段都要有明确的主题和与整体的关联
- 语言要有节奏感，长短句结合
- 展现真实的思考过程和情感变化"""

            if context and context.get("major"):
                system_prompt += f"\n\n学生申请的专业是{context['major']}，请确保内容与该专业高度相关。"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4000,
                temperature=0.8,  # 增加一点创造性
                presence_penalty=0.2,  # 避免重复
                frequency_penalty=0.2  # 鼓励多样性
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            return {
                "content": content,
                "model": self.model,
                "tokens_used": usage.total_tokens,
                "cost": (usage.total_tokens / 1000) * 0.02,
                "success": True
            }
        except Exception as e:
            return {
                "content": f"OpenAI API error: {str(e)}",
                "success": False,
                "error": str(e)
            }

class ClaudeClient:
    def __init__(self):
        if ANTHROPIC_API_KEY:
            self.client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            self.model = "claude-3-sonnet-20240229"
            self.available = True
        else:
            self.client = None
            self.available = False
    
    async def generate_content(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.available:
            return {"content": "Claude API key not configured", "success": False, "error": "No API key"}
        
        try:
            # Claude更擅长结构化思维和学术写作
            system_prompt = """你是一位专业的学术文书专家，擅长帮助学生撰写高质量的个人陈述。你的专长包括：

1. 构建清晰的逻辑结构和论证链条
2. 平衡学术深度与可读性
3. 精准把握不同专业的申请要点
4. 创作真实自然、有说服力的叙述

写作原则：
- 用具体事例和数据支撑观点
- 展现申请者的思考深度和成长轨迹
- 保持专业性的同时体现个人特色
- 避免空洞的描述和AI化的表达
- 确保每段都为整体论证服务"""

            if context and context.get("major"):
                system_prompt += f"\n\n申请专业：{context['major']}。请确保内容体现该专业所需的关键素质和能力。"
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            usage = response.usage
            
            return {
                "content": content,
                "model": self.model,
                "tokens_used": usage.input_tokens + usage.output_tokens,
                "cost": (usage.input_tokens / 1_000_000) * 3.0 + (usage.output_tokens / 1_000_000) * 15.0,
                "success": True
            }
        except Exception as e:
            return {
                "content": f"Claude API error: {str(e)}",
                "success": False,
                "error": str(e)
            }

class DeepSeekClient:
    def __init__(self):
        if DEEPSEEK_API_KEY:
            self.api_key = DEEPSEEK_API_KEY
            self.base_url = "https://api.deepseek.com/v1"
            self.model = "deepseek-chat"
            self.available = True
        else:
            self.api_key = None
            self.available = False
    
    async def generate_content(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.available:
            return {"content": "DeepSeek API key not configured", "success": False, "error": "No API key"}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            system_content = "你是一位专业的文书写作助手，擅长创作自然、有说服力的个人陈述。请用具体的细节和真实的语言风格来写作，避免AI化的表达。"
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.7,
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
                    "cost": (total_tokens / 1000) * 0.001,
                    "success": True
                }
        except Exception as e:
            return {
                "content": f"DeepSeek API error: {str(e)}",
                "success": False,
                "error": str(e)
            }

# 模型工厂
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
        if MOCK_MODE or model_type == ModelType.MOCK:
            return await self._generate_mock_content(prompt, context)
        
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
    
    async def _generate_mock_content(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate mock content for testing"""
        major = context.get("major", "计算机科学") if context else "计算机科学"
        
        # 根据prompt内容判断生成什么类型的内容
        if "大纲" in prompt or "outline" in prompt.lower():
            content = self._generate_mock_outline(major)
        elif "文书" in prompt or "personal statement" in prompt.lower():
            content = self._generate_mock_essay(major)
        else:
            content = f"这是关于{major}专业的模拟AI响应内容。在实际应用中，这里会调用真实的AI模型生成个性化内容。"
        
        return {
            "content": content,
            "model": "mock-model",
            "tokens_used": len(content),
            "cost": 0.0,
            "success": True
        }
    
    def _generate_mock_outline(self, major: str) -> str:
        """生成高质量的模拟大纲"""
        if "城市管理" in major or "城市规划" in major or "urban" in major.lower():
            return f"""
# {major}个人陈述大纲

## 第一段：引言 - "城市的温度"
**核心画面**：从一次城市实地调研中的具体观察开始
- 可能的开头：在某个城市角落发现的细节（如老旧社区的适老化改造、智慧交通系统的人性化设计等）
- 引出核心问题：技术如何让城市变得更"聪明"，同时保持人文温度？
- **衔接**：这个观察如何引发了你对城市管理的深度思考

## 第二段：学术基础 - "工具与视野的建立"
**重点**：不是简单罗列课程，而是展现认知发展轨迹
- 从基础课程中获得的理论框架（如城市规划理论、公共管理等）
- 通过具体的课程项目或研究，展现分析城市问题的方法论
- **关键**：强调数据分析、GIS等技术工具如何帮你"看见"城市的复杂性
- **衔接**：理论学习如何为实践探索奠定基础

## 第三段：实践探索一 - "发现问题"
**建议场景**：第一次真正深入城市实践的经历
- 具体的项目背景和你的角色
- 遇到的挑战和意外发现
- 运用技术工具解决实际问题的过程
- **收获**：对城市管理复杂性的初步认识
- **衔接**：这次经历如何驱动你寻求更深层的解决方案

## 第四段：实践探索二 - "深化理解"
**建议角度**：更具挑战性的项目或跨领域合作
- 与第三段形成对比或递进关系
- 展现更成熟的问题分析和解决能力
- 体现跨学科思维或创新方法的运用
- **重点**：你如何平衡效率与公平、技术与人文
- **衔接**：实践经验如何塑造了你的专业理想

## 第五段：未来规划 - "愿景与行动"
**结构**：Why School + Career Goals + Personal Vision
- **Why this program**：具体说明项目的哪些特色与你的兴趣匹配
- **短期目标**：毕业后3-5年的具体职业规划
- **长期愿景**：你希望在城市管理领域产生什么样的影响
- **回扣主题**：重新呼应第一段的核心关切

**整体主线**：从观察城市问题 → 建立分析框架 → 实践探索解决方案 → 形成专业理想
**避免**：空洞的概念堆砌、千篇一律的表达、缺乏个人特色的描述
            """
        
        elif "计算机" in major or "computer" in major.lower():
            return f"""
# {major}个人陈述大纲

## 第一段：引言 - "代码背后的思考"
**开头建议**：从一次编程经历中的"顿悟时刻"开始
- 具体场景：可能是debugging过程中的发现，或是看到代码解决实际问题的瞬间
- 引出思考：技术不仅是工具，更是连接想法与现实的桥梁
- **核心问题**：如何用技术创造有意义的改变？

## 第二段：学术基础 - "从好奇到专精"
**重点**：展现学习的深度和广度
- 核心课程学习和理论基础的建立
- 通过具体项目展现技术能力的发展
- **关键**：不只是技术技能，更是计算思维和问题解决能力
- **衔接**：学术基础如何支撑实践探索

## 第三段：技术实践 - "解决真实问题"
**建议内容**：最有代表性的技术项目
- 项目背景和技术挑战
- 你的创新解决方案和实现过程
- 项目的impact和个人收获
- **亮点**：体现你的技术判断力和工程能力

## 第四段：跨界探索 - "技术的边界"
**建议角度**：技术与其他领域的结合
- 可能是跨学科项目、实习经历或社会实践
- 展现技术在更广阔场景中的应用
- **重点**：你对技术社会影响的思考

## 第五段：未来愿景 - "下一个章节"
**包含**：Why School + Career Goals + Long-term Vision
- 为什么选择这个项目和学校
- 具体的学习目标和研究兴趣
- 职业规划和长远影响力目标

**整体主线**：技术兴趣萌发 → 能力体系建立 → 实践应用探索 → 社会价值思考 → 未来发展规划
            """
        
        else:
            return f"""
# {major}个人陈述大纲

## 第一段：引言 - 兴趣起源
- 描述您对{major}领域产生兴趣的关键时刻
- 用具体的场景或经历开始
- 提出您想要深入探索的核心问题

## 第二段：学术准备
- 相关课程学习和理论基础
- 通过具体项目展现学习成果
- 为专业发展奠定的知识基础

## 第三段：实践经历一
- 第一个重要的相关实践经历
- 具体的挑战和解决过程
- 获得的技能和认知

## 第四段：实践经历二
- 更深入或更具挑战性的经历
- 展现成长和能力提升
- 与专业目标的连接

## 第五段：未来规划
- Why School和项目选择理由
- 明确的学习目标和职业规划
- 长期的专业愿景

**主线**：兴趣发现 → 知识积累 → 实践应用 → 深度探索 → 未来发展
            """
    
    def _generate_mock_essay(self, major: str) -> str:
        """生成高质量的模拟文书"""
        if "城市管理" in major or "城市规划" in major or "urban" in major.lower():
            return """
我第一次看见一座城市“发烧”，是通过一张热力图。在地理信息系统规划应用的课堂，我用ArcGIS对北京市的微博打卡数据进行核密度分析，发现一些热门文化地标被巨量人流“点燃”，而周边的居民区却在数据中逐渐“降温”：它们不再是城市关注的焦点。这总让我担忧，当城市治理越发依赖数据，那些市井烟火以及那些寻常人的需求是否还能被看见。后来在苏州工业园实习，指尖抚过厂房外斑驳的小广告时，浆糊残留的颗粒感突然扎醒了我：数据扫不到的墙角，正藏着某个打工者找活计的期盼。现在每当我打开数据分析软件，总想起那些被算法判定为低热度区域的角落。如何让算法识别菜市场的轮椅坡道？怎样在交通模型中为老街坊晒太阳留出空间？这些疑问推着我往数字城市管理的路上走，毕竟真正的智慧城市，不该是WIFI满格却人情断连的地方。

在北京大学学习城乡规划的五年里，我学会了用数据工具丈量城市的复杂度。在区域分析与区域规划课程中，我通过AHP层次分析法对服务业密度、交通条件、公共设施可达性等因子建模评估，尝试还原居民日常出行与空间选择的逻辑；而在珠三角生态补偿机制研究中，我结合问卷调研与空间计量模型，构建了九市四十八个县区的生态支付意愿指数，发现经济发展与环保意愿之间并非简单的线性关系，而需结合地方财政、产业结构和文化认知共同考量。数据的价值不止于预测，更在于解释。我渐渐意识到，学做一个规划师，不是靠一张完美的图纸，而是靠对城市未来各种“不确定性”的理解和准备；而数据，是帮助我看见这些可能性的工具。

带着这样的思考，我在工程设计实践中摸索数据与人文的接口。2023年暑期，我参与了江苏盐业井神楚州储气库一期地面工程的设计工作。这是一个偏离典型城市核心区的能源基础设施项目，借此机会，我开始思考：在这些被算法标注为非活跃区域的角落，应该如何考虑人的存在与体验？项目之初，我们面对的是18%的地面高差、交错复杂的地下管线，以及来自工艺、结构、景观三方截然不同的需求。我通过BIM工具整合地质勘测数据与客户功能要求，构建三维模型并协助推进多轮方案。在一次交叉评审中，工艺部门提出压缩办公区面积以扩大设备间距，我借助人流模拟展示了此方案在通勤路径和应急疏散上的隐患，最终说服多方保留了厂区中央的公共景观节点。这个决定或许在生产效率上让位半步，却为日后这里的管理人员与一线员工留下了一片可以喘息和交流的开放空间。所谓“智慧”，它不一定是更复杂的算法，而是在标准化流程中找到为人留白的空间。哪怕只是一条多走三秒的人行路径、一个能坐下喝水的绿地节点，也能让一座封闭的工业设施多一分温度。

这种对“空间温度”的敏感，也贯穿在我后续的实践中。2023年秋，我参与了景迈山古茶林文化景观世界文化遗产申报项目，担任景观规划实习生。为了还原村落真实的空间使用情况，我参与建立了200多栋传统建筑的数据库，并运用GIS技术分析空间分布特征，识别出民宿扩张、公共设施不足等主要问题。面对生态敏感区内的违规建设，我叠加航拍图和建筑数据，绘制出热力图，并据此参与编写《景迈山建设管控技术导则》。其中，我设计了一个结合地形坡度和阳光照射的限高模型，用以保护茶树的光照需求。但真正的挑战，是如何让村民理解这些技术规定。有村民提出采茶季需要临时搭建宿舍，也有人希望保留靠近茶林的猪圈，以便兼顾生计与日常照料。我改用BIM模型结合方言讲解，才逐步建立信任，推动43份建房方案中92%完成调整。由此可见，真正有效的空间治理，不是单方面制定标准，而是回应人们真实的生活需求。

对“空间温度”的追问，还引导我关注日常生活中最基础的空间分配，即居住的公平与尊严。我参与了关于新加坡高密度住房宜居性的研究，思考在有限空间中如何兼顾效率与公平。为了评估不同组屋项目的宜居性，我在大巴窑、淡滨尼等四个典型组屋区开展调研，结合实地观察与居民访谈，从居住环境、配套设施、交通便利度等维度建立评价指标。通过对比，我发现高效率带来的标准化设计虽然提升了空间利用率，但也可能削弱地方文化与日常生活的多样性。例如，一些新建组屋虽采光良好、设施齐全，却缺少邻里互动空间，影响了居民的归属感。调研最后，我尝试提出一套结合家庭收入、通勤需求与教育资源分布的住房分配建议模型，探索如何通过数据建模提升分配机制的公平性。空间规划不仅是对物理结构的设计，更是对社会结构的干预；智慧城市的理想状态，应该是让每一个人都能在合适的位置上被安放和理解。

我希望成为构建这种安放机制的参与者，让数据决策与人文关怀并行。因此，毕业后我计划加入城市规划机构、政府部门或专注于城市发展的咨询公司，从社区参与规划、可持续住房项目或智慧城市应用等方向入手，支持城市更新决策。长期而言，我希望能在智库或公共管理系统内工作，推动面向人本需求的数字治理机制设计。要实现这样的目标，我仍需在技术整合、算法建模方面补强。因此，港大的Master of Digital Urban Management项目吸引了我，尤其是其Urban Technology and Analytics方向。相比我在本科阶段以GIS和空间分析为主的训练，港大课程更加强调数据驱动治理所需的模型与数据整合能力。例如， Urban Big Data Analytics与Programming and Foundations in Urban Data Analysis将帮助我建立面向复杂城市问题的技术解决方案；Artificial Intelligence for Future Cities则让我能在智慧社区、住房算法分配等领域探索智能决策与社会公正的平衡点。我始终相信，智慧城市的未来，不该只是更快的算法，而是更懂人的城市。
            """
  
        elif "计算机" in major or "computer" in major.lower():
            return """
我的代码启蒙带着咸涩的海风味。在靠海的渔村长大，我的童年没有键盘，十五岁在舅舅家发黄的《电脑爱好者》杂志上邂逅的“Hello World”，像渔火点亮了夜海。大学勤工俭学买的二手笔记本，承载了我开发的第一个农业GIS系统；荔枝FM工作室调试的H5小游戏和音频平台，教会我代码如何传递温度；辗转阿里、腾讯、字节等大厂担任前端工程师，我一步步积累了宝贵的工程实践经验。近几年开始，在工作场景中AI与前端深度融合的背景下，我意识到技术变革带来的挑战与机遇。边缘计算和浏览器端AI模型的兴起让前端领域发生翻天覆地的变化。要在这波技术浪潮中继续发挥价值，我希望更新我的知识积累。这正是我申请东北大学CS项目的原因：通过这次深造，突破职业天花板，将我拥有的专业知识与系统工程和AI领域的训练相结合，为未来的技术变革做好准备。

我的职业道路有几个里程碑式的时刻，其中最难忘的发生在毕业后第二年加入阿里巴巴期间。2016年 “双11”购物节当天，天猫手机客户端突然出现大面积白屏现象，每秒损失的交易额以百万计算。作为值班团队成员，面对上亿用户无法正常购物的紧急情况，我第一次感受到了真正的技术压力。在键盘急促的敲击声中，我想起本科期间积累的知识：Object Oriented Programming中学到的异常处理策略和模块化思维以及Operating System中掌握的资源管理理念。我冷静地使用前端调试工具层层深入，最终定位到一个未被捕获的异常和一处边缘条件处理缺陷。这些代码瑕疵在巨大流量冲击下成为了致命弱点。修复问题并紧急发布更新后，我不仅获得了当年的高绩效评价，更重要的是领悟到技术的本质：当理论与实践融为一体时，才能真正应对产业环境中的挑战。从此，我开始更加重视代码质量和系统健壮性，这也成为贯穿我整个职业生涯的核心准则。

从这次危机处理中汲取的经验，如同一粒种子，在我来到腾讯后迎来了生长的契机。2018年，当我已经习惯了前端页面的开发维护工作时，全新的挑战悄然而至：我们部门需要建立一个自动化平台，捕捉各类媒体信息进行舆情分析。这不再是单纯的前端工作，而是需要全栈能力的项目：从页面展示到数据抓取、存储和处理的完整链路。为了迅速提升后端能力，我利用工作之余自学Redis用于高速缓存、Elasticsearch实现全文检索、Node.js构建服务端应用，以及必要的Linux运维知识。在项目开发过程中，我分析了项目的实际需求和瓶颈所在。通过对比多种技术路径，我带领团队最终设计了一套分布式数据处理架构，实现高效稳定地抓取大量网页内容。经过近一年的不懈努力和持续迭代，平台最终高质量交付，为公司的舆情分析提供了可靠的数据基础。这个成果不仅获得了内部的认可，还有幸在GIAC全球互联网架构大会上分享。那时，我清晰地意识到：技术的价值不在于掌握多少独立的工具，而在于如何将这些工具有机地组合，解决实际业务问题。从前端到全栈的转变，不只是技术广度的扩展，更是思维方式的进化，我开始以更整体的视角看待软件工程。

腾讯的经历为我打开了新视野，自2020年加入字节跳动后，我经历了更全面的技术与管理双重挑战。入职伊始，我被委以重任——独立管理两个低代码平台，带领一支由实习生、外包和正式员工组成的混合团队开发JIMU低代码项目，打造出既能让5万名员工轻松创建内容审核页面，又不会在高负载下崩溃的平台。这个项目让我对系统架构产生了更深的思考，也让我从技术专家转变为项目协调者和团队引导者。于是，我主动增加与上级的一对一沟通，向他们学习项目管理经验；同时，我开始研究如何更有效地分解任务和分配资源。当JIMU平台同时面临外部交付压力和内部技术升级需求时，我通过有理有据的沟通，成功争取到额外资源支持外部需求；同时，我设计了内部技术改造的任务分配方案，确保每位团队成员既能在擅长领域充分发挥，又不会感到过度负荷。这种平衡的管理方式最终让平台超预期完成了交付目标，同时实现了从低效率、高人工干预到高效率、低干预的质的飞跃。

我的职业旅程是一幅逐渐清晰的拼图，每一片都铭刻着成长与蜕变。在阿里巴巴，我意识到代码质量的重要性；在腾讯，我认识到技术广度的价值；而在字节跳动，我深刻体会到技术与管理相结合的挑战。如今，站在职业发展的新十字路口。一方面，我看到了技术领域正在发生的根本性变革，不再是简单的工具更迭，而是计算范式的转变。大型语言模型正在重塑前端开发流程，这些变化远超过往的技术迭代。另一方面，作为一名逐渐承担更多领导责任的工程师，我需要更系统的理论知识来支撑技术决策，更全面的项目经验来提升团队管理能力。NEU的Computer Science硕士项目恰如我所需。课程如Building Scalable Distributed Systems将帮助我构建大规模系统的理论基础；Advanced Machine Learning能让我深入理解AI前沿技术；备受业界认可的co-op项目则提供了将理论应用于实践的宝贵机会。我尤其期待通过团队项目磨练协作能力，这对未来带领更大规模团队至关重要。技术给了我改变命运的力量。通过深造，我不仅追求个人的职业突破，更希望能在AI与传统软件工程融合的浪潮中，用所学知识为行业创造更大价值，鼓励像我一样来自基层的年轻人。
            """
        else:
            return f"""
这里是一篇关于{major}专业的高质量个人陈述示例。在实际应用中，系统会根据用户的具体经历和材料生成个性化的内容。

该文书将包含：
- 生动的开头场景，展现对专业的兴趣起源
- 扎实的学术基础介绍，体现学习能力
- 2-3个具体的实践经历，展现能力发展轨迹  
- 明确的未来规划和Why School
- 自然的语言风格，避免AI化表达

实际生成的文书会基于用户提供的真实经历，创造出独特且有说服力的申请材料。
            """
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        if MOCK_MODE:
            return [ModelType.MOCK]
        return [model for model, client in self._clients.items() if getattr(client, 'available', False)]

# 智能模型选择器
class EnhancedModelSelector:
    def select_model(self, step: str, context: Dict[str, Any], preferences: Dict[str, Any] = None) -> ModelType:
        """智能选择最适合的模型"""
        if MOCK_MODE:
            return ModelType.MOCK
        
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
        if any(term in major for term in ["computer", "计算机", "software", "cs", "tech"]):
            if step in ["step5", "step7"]:
                return ModelType.CLAUDE_SONNET if step == "step5" else ModelType.OPENAI_GPT4
        
        return base_preferences.get(step, ModelType.DEEPSEEK)

# 存储管理器
class StorageManager:
    def __init__(self):
        self.redis = redis_client
        self.use_redis = redis_client is not None
    
    def set_data(self, key: str, value: str, expiry: int = SESSION_EXPIRY):
        """存储数据"""
        if self.use_redis:
            self.redis.setex(key, expiry, value)
        else:
            memory_storage[key] = {
                "value": value,
                "expiry": time.time() + expiry
            }
    
    def get_data(self, key: str) -> Optional[str]:
        """获取数据"""
        if self.use_redis:
            return self.redis.get(key)
        else:
            if key in memory_storage:
                data = memory_storage[key]
                if time.time() > data["expiry"]:
                    del memory_storage[key]
                    return None
                return data["value"]
            return None
    
    def delete_data(self, key: str):
        """删除数据"""
        if self.use_redis:
            self.redis.delete(key)
        else:
            memory_storage.pop(key, None)

# 会话管理器
class SessionManager:
    def __init__(self):
        self.storage = StorageManager()
    
    async def create_session(self, user_id: Optional[str] = None) -> str:
        """创建新的会话"""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "current_step": "step1",
            "created_at": datetime.now().isoformat(),
            "materials": {},
            "model_history": [],
            "cost_tracking": {"total_tokens": 0, "estimated_cost": 0.0, "calls": []}
        }
        
        await self.save_session(session_id, session_data)
        return session_id
    
    async def save_session(self, session_id: str, data: Dict[str, Any]):
        """保存会话数据"""
        key = f"session:{session_id}"
        self.storage.set_data(key, json.dumps(data))
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话数据"""
        key = f"session:{session_id}"
        data = self.storage.get_data(key)
        return json.loads(data) if data else None
    
    async def delete_session(self, session_id: str):
        """删除会话"""
        key = f"session:{session_id}"
        self.storage.delete_data(key)

# 缓存管理器
class CacheManager:
    def __init__(self):
        self.storage = StorageManager()
    
    async def get_cached_response(self, prompt: str, model: str, context: Dict[str, Any]) -> Optional[str]:
        """获取缓存的响应"""
        import hashlib
        context_hash = str(hash(json.dumps(context, sort_keys=True)))
        content = f"{prompt}:{model}:{context_hash}"
        cache_key = f"cache:response:{hashlib.md5(content.encode()).hexdigest()}"
        
        cached = self.storage.get_data(cache_key)
        if cached:
            cached_data = json.loads(cached)
            return cached_data.get("response")
        return None
    
    async def cache_response(self, prompt: str, model: str, context: Dict[str, Any], response: str):
        """缓存响应"""
        import hashlib
        context_hash = str(hash(json.dumps(context, sort_keys=True)))
        content = f"{prompt}:{model}:{context_hash}"
        cache_key = f"cache:response:{hashlib.md5(content.encode()).hexdigest()}"
        
        cache_data = {
            "response": response,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "context_hash": context_hash
        }
        
        self.storage.set_data(cache_key, json.dumps(cache_data), CACHE_EXPIRY)

# 成本追踪器
class CostTracker:
    def __init__(self):
        self.storage = StorageManager()
        self.daily_budget = float(os.getenv("DAILY_API_BUDGET", 50.0))
    
    async def track_usage(self, session_id: str, model: str, tokens: int, cost: float):
        """记录API使用情况"""
        if MOCK_MODE:
            return  # 模拟模式不记录成本
        
        # 记录到会话
        session_manager = SessionManager()
        session_data = await session_manager.get_session(session_id)
        if session_data:
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
            
            await session_manager.save_session(session_id, session_data)
    
    async def check_budget_limit(self) -> Dict[str, Any]:
        """检查是否超出预算限制"""
        if MOCK_MODE:
            return {
                "daily_budget": self.daily_budget,
                "used_today": 0.0,
                "remaining": self.daily_budget,
                "exceeded": False,
                "usage_percentage": 0.0
            }
        
        today = datetime.now().strftime("%Y-%m-%d")
        daily_key = f"usage:daily:{today}"
        daily_data = self.storage.get_data(daily_key)
        
        if daily_data:
            daily_stats = json.loads(daily_data)
            current_cost = daily_stats.get("total_cost", 0.0)
        else:
            current_cost = 0.0
        
        remaining_budget = self.daily_budget - current_cost
        
        return {
            "daily_budget": self.daily_budget,
            "used_today": current_cost,
            "remaining": remaining_budget,
            "exceeded": current_cost > self.daily_budget,
            "usage_percentage": (current_cost / self.daily_budget) * 100
        }

# 全局实例
model_factory = ModelFactory()
model_selector = EnhancedModelSelector()
session_manager = SessionManager()
cache_manager = CacheManager()
cost_tracker = CostTracker()

# 文书创作步骤定义
STEPS = {
    "step1": {
        "name": "基本信息收集",
        "prompt": "欢迎使用StoryScape AI文书助手！🎓\n\n让我来帮助您创建一份出色的个人陈述。首先，请告诉我：\n\n📚 **申请专业**：您要申请什么专业？\n🏫 **目标学校**：有特定的目标学校吗？（可选）\n\n请这样格式回复：专业名称，学校名称\n例如：计算机科学，斯坦福大学",
        "requires_ai": False
    },
    "step2": {
        "name": "材料收集",
        "prompt": "很好！现在请尽可能详细地提供您的背景材料。这些真实的经历将是您文书的基石：\n\n📚 **教育背景**：\n- 学校、专业、GPA或排名\n- 印象最深刻的课程及具体收获\n- 课程项目、研究经历的详细描述\n\n💼 **实习/工作经历**：\n- 公司/机构名称、职位、具体工作内容\n- 遇到的挑战和解决方案\n- 具体的成果和数据（如提升XX%，完成XX项目）\n\n🚀 **项目经历**：\n- 项目背景、你的角色、技术栈\n- 具体遇到的问题和创新解决方案\n- 项目影响和个人收获\n\n🏆 **获奖/成就**：\n- 奖项名称、获奖原因、竞争激烈程度\n- 其他能体现能力的具体成就\n\n🎯 **课外活动/社团**：\n- 活动性质、你的角色、具体贡献\n- 体现领导力或团队协作的经历\n\n💡 **其他特殊经历**：\n- 志愿服务、交换学习、特殊兴趣爱好\n- 任何塑造你价值观或影响你选择的经历\n\n**请尽可能详细**，包含具体的时间、地点、数字、对话等细节。这些细节将帮助AI为您创造更生动、更有说服力的故事。",
        "requires_ai": False
    },
    "step3": {
        "name": "核心经历提取",
        "prompt": "感谢您提供的详细材料！👏\n\n现在请从您所有的经历中，选择**2-3个最重要的核心经历**。这些经历将成为您文书的主线：\n\n🎯 **选择标准**：\n- 最能体现您的能力、性格和价值观的经历\n- 与申请专业最相关的经历  \n- 对您产生重要影响或改变的经历\n- 有具体故事可讲的经历（有冲突、有解决过程、有收获）\n\n📝 **对于每个选择的经历，请提供**：\n- **具体背景**：什么时候、在哪里、什么情况下\n- **核心挑战**：您遇到了什么具体的问题或困难\n- **解决过程**：您是如何思考和行动的\n- **具体成果**：产生了什么impact，学到了什么\n- **深层思考**：这个经历如何影响了您对专业/未来的看法\n\n💡 **写作提示**：\n- 用具体的场景、对话、细节来描述\n- 体现您的思考过程和成长轨迹\n- 展现您解决问题的能力和方法\n\n这些核心经历将帮我们构建一个连贯、有说服力的个人成长故事。",
        "requires_ai": False
    },
    "step4": {
        "name": "动机和目标",
        "prompt": "完美！最后的准备工作：🎯\n\n请分享您申请这个专业的深层动机和未来规划：\n\n💡 **专业兴趣起源**：\n- 什么具体的经历、时刻或发现让您对这个专业产生兴趣？\n- 有没有某本书、某个项目、某次实习让您\"眼前一亮\"？\n- 是渐进式的兴趣发展，还是某个转折点？\n\n📖 **学习目标**：\n- 您最想在研究生阶段深入学习什么？\n- 有哪些具体的研究方向或课程让您特别期待？\n- 您觉得自己目前还缺少哪些知识或技能？\n\n🚀 **职业规划**：\n- **短期目标**（毕业后1-3年）：希望在什么行业、什么类型的公司/机构工作？\n- **长期愿景**（5-10年）：希望在这个领域达到什么位置或产生什么影响？\n- 有没有特别想解决的社会问题或想达成的职业理想？\n\n🏫 **Why School**（如有目标学校）：\n- 为什么选择这所学校的这个项目？\n- 哪些具体的课程、教授、资源或机会吸引您？\n- 您觉得自己的背景和这个项目如何匹配？\n\n请用真实、具体的语言描述您的想法。这些将帮助我们打造一个令人信服的\"为什么要录取您\"的故事。",
        "requires_ai": False
    },
    "step5": {
        "name": "大纲生成",
        "prompt": "正在基于您的材料生成个性化的文书大纲...",
        "requires_ai": True,
        "ai_prompt_template": ENHANCED_PROMPTS["outline_generation"]
    },
    "step6": {
        "name": "大纲确认",
        "prompt": "请查看上面生成的大纲，并提供您的反馈：\n\n✅ **确认满意的部分**：哪些部分很好地捕捉了您的经历和想法？\n\n🔄 **需要调整的部分**：\n- 哪些段落的重点需要调整？\n- 有没有想要强调的经历没有被突出？\n- 段落顺序是否合理？\n- 有没有想要改变的故事线索或主题？\n\n💡 **补充建议**：\n- 有没有重要的细节、经历或想法没有被包含？\n- 您希望在文书中体现什么样的个人特质？\n- 对语言风格有什么偏好（更学术、更活泼、更严谨等）？\n\n如果您对大纲整体满意，请回复**\"确认大纲\"**，我们将进入文书写作阶段。\n\n如果需要修改，请具体说明您的要求，我会调整大纲。",
        "requires_ai": False
    },
    "step7": {
        "name": "文书生成",
        "prompt": "正在基于确认的大纲撰写您的个人陈述...",
        "requires_ai": True,
        "ai_prompt_template": ENHANCED_PROMPTS["essay_generation"]
    },
    "step8": {
        "name": "最终完善",
        "prompt": "请查看上面生成的个人陈述，并提供您的反馈：\n\n📝 **内容方面**：\n- 哪些部分需要增加、删除或修改？\n- 有没有想要强调的细节或经历没有体现好？\n- 故事线是否清晰连贯？\n\n🎨 **语言风格**：\n- 哪些表达可以更自然或更有力？\n- 有没有听起来像AI写的句子需要改进？\n- 语言风格是否符合您的个人特色？\n\n🔍 **结构完善**：\n- 段落间的衔接是否自然？\n- 开头和结尾是否相呼应？\n- 重点分配是否合适？\n\n⚖️ **目标匹配**：\n- 是否充分展现了您与申请专业的匹配度？\n- 有没有突出您的独特优势？\n\n请提供具体的修改建议，我会据此优化文书。\n\n如果您对当前版本满意，请回复**\"确认最终版本\"**。",
        "requires_ai": False
    }
}

@app.get("/")
async def root():
    return {"message": "StoryScape Enhanced API is running", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        if redis_client:
            redis_client.ping()
            redis_status = "connected"
        else:
            redis_status = "memory_mode"
        
        return {
            "status": "healthy",
            "redis": redis_status,
            "mock_mode": MOCK_MODE,
            "available_models": len(model_factory.get_available_models()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/api/start", response_model=SessionResponse)
async def start_session(request: SessionRequest):
    """开始新的文书创作会话"""
    try:
        session_id = await session_manager.create_session(request.user_id)
        
        return SessionResponse(
            session_id=session_id,
            current_step="step1",
            prompt=STEPS["step1"]["prompt"],
            model_used="system"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process", response_model=SessionResponse)
async def process_step(request: ProcessRequest, background_tasks: BackgroundTasks):
    """处理用户响应并推进到下一步"""
    start_time = time.time()
    
    # 获取会话数据
    session_data = await session_manager.get_session(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    current_step = session_data["current_step"]
    user_response = request.response.strip()
    
    if not user_response:
        raise HTTPException(status_code=400, detail="Response cannot be empty")
    
    try:
        # 处理当前步骤
        result = await process_current_step(session_data, current_step, user_response)
        
        # 更新会话数据
        session_data.update(result["session_updates"])
        await session_manager.save_session(request.session_id, session_data)
        
        processing_time = time.time() - start_time
        
        return SessionResponse(
            session_id=request.session_id,
            current_step=result["next_step"],
            prompt=result["prompt"],
            generated_content=result.get("generated_content"),
            model_used=result.get("model_used"),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

async def process_current_step(session_data: Dict[str, Any], current_step: str, user_response: str) -> Dict[str, Any]:
    """处理当前步骤的逻辑"""
    
    if current_step == "step1":
        # 解析专业和学校信息
        parts = [p.strip() for p in user_response.split(',')]
        major = parts[0] if parts else user_response
        school = parts[1] if len(parts) > 1 else ""
        
        return {
            "next_step": "step2",
            "prompt": STEPS["step2"]["prompt"],
            "session_updates": {
                "major": major,
                "school": school,
                "materials": {"basic_info": user_response}
            }
        }
    
    elif current_step == "step2":
        return {
            "next_step": "step3", 
            "prompt": STEPS["step3"]["prompt"],
            "session_updates": {
                "materials": {**session_data.get("materials", {}), "background": user_response}
            }
        }
    
    elif current_step == "step3":
        return {
            "next_step": "step4",
            "prompt": STEPS["step4"]["prompt"], 
            "session_updates": {
                "materials": {**session_data.get("materials", {}), "highlights": user_response}
            }
        }
    
    elif current_step == "step4":
        # 收集动机后，生成大纲
        session_data["materials"]["motivation"] = user_response
        
        # 选择模型并生成大纲
        selected_model = model_selector.select_model("step5", session_data)
        context = {**session_data, "session_id": session_data["session_id"]}
        outline = await generate_ai_content("step5", session_data, selected_model, context)
        
        return {
            "next_step": "step6",
            "prompt": STEPS["step6"]["prompt"],
            "generated_content": outline,
            "model_used": selected_model.value,
            "session_updates": {
                "materials": {**session_data.get("materials", {}), "motivation": user_response},
                "outline": outline,
                "current_step": "step6"
            }
        }
    
    elif current_step == "step6":
        # 处理大纲反馈并生成文书
        if "确认大纲" in user_response.lower():
            outline_feedback = "用户确认大纲无需修改"
        else:
            outline_feedback = user_response
        
        selected_model = model_selector.select_model("step7", session_data)
        context = {**session_data, "session_id": session_data["session_id"]}
        session_data["outline_feedback"] = outline_feedback
        
        draft = await generate_ai_content("step7", session_data, selected_model, context)
        
        return {
            "next_step": "step8",
            "prompt": STEPS["step8"]["prompt"],
            "generated_content": draft,
            "model_used": selected_model.value,
            "session_updates": {
                "outline_feedback": outline_feedback,
                "draft": draft,
                "current_step": "step8"
            }
        }
    
    elif current_step == "step8":
        # 最终完善
        if "确认最终版本" in user_response.lower():
            final_feedback = "用户确认最终版本"
        else:
            final_feedback = user_response
        
        # 可选：基于反馈进一步优化文书
        final_draft = session_data.get("draft", "")
        if "确认最终版本" not in user_response.lower():
            # 这里可以调用AI进行最终修改
            selected_model = model_selector.select_model("step8", session_data)
            context = {**session_data, "session_id": session_data["session_id"]}
            final_draft = await refine_draft(session_data.get("draft", ""), final_feedback, selected_model, context)
        
        return {
            "next_step": "complete",
            "prompt": "🎉 您的个人陈述已完成！\n\n您可以：\n- 下载最终版本\n- 继续微调\n- 开始新的文书创作\n\n感谢使用StoryScape！",
            "generated_content": final_draft,
            "session_updates": {
                "final_feedback": final_feedback,
                "final_draft": final_draft,
                "completed_at": datetime.now().isoformat(),
                "current_step": "complete"
            }
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Invalid step: {current_step}")

async def generate_ai_content(step: str, session_data: Dict[str, Any], model: ModelType, context: Dict[str, Any]) -> str:
    """生成AI内容的核心函数"""
    
    # 检查缓存
    prompt_template = STEPS[step]["ai_prompt_template"]
    materials = session_data.get("materials", {})
    
    # 构建完整的prompt - 使用增强的模板
    if step == "step5":  # 大纲生成
        full_prompt = prompt_template.format(
            major=session_data.get("major", ""),
            school=session_data.get("school", ""),
            background=materials.get("background", ""),
            highlights=materials.get("highlights", ""),
            motivation=materials.get("motivation", "")
        )
    elif step == "step7":  # 文书生成
        full_prompt = prompt_template.format(
            outline=session_data.get("outline", ""),
            outline_feedback=session_data.get("outline_feedback", ""),
            major=session_data.get("major", ""),
            school=session_data.get("school", ""),
            background=materials.get("background", ""),
            highlights=materials.get("highlights", ""),
            motivation=materials.get("motivation", "")
        )
    else:
        full_prompt = prompt_template
    
    # 检查缓存
    cached_response = await cache_manager.get_cached_response(
        full_prompt, model.value, {"session_context": session_data.get("major", "")}
    )
    
    if cached_response:
        return cached_response
    
    # 调用AI模型
    response = await call_ai_model(model, full_prompt, context)
    
    # 缓存响应
    await cache_manager.cache_response(
        full_prompt, model.value, {"session_context": session_data.get("major", "")}, response
    )
    
    return response

async def call_ai_model(model: ModelType, prompt: str, context: Dict[str, Any] = None) -> str:
    """调用具体的AI模型"""
    
    # 检查预算限制
    budget_status = await cost_tracker.check_budget_limit()
    if budget_status["exceeded"]:
        if model != ModelType.DEEPSEEK:
            print(f"Budget exceeded, switching to DeepSeek for cost efficiency")
            model = ModelType.DEEPSEEK
    
    # 使用系统prompt增强模型表现
    enhanced_prompt = f"{ENHANCED_PROMPTS['system_base']}\n\n{prompt}"
    
    # 调用模型工厂
    result = await model_factory.generate_content(model, enhanced_prompt, context)
    
    if result["success"]:
        # 记录使用情况和成本
        session_id = context.get("session_id", "unknown") if context else "unknown"
        await cost_tracker.track_usage(
            session_id=session_id,
            model=result["model"],
            tokens=result["tokens_used"], 
            cost=result["cost"]
        )
        
        return result["content"]
    else:
        # 如果主模型失败，尝试备用模型
        if model != ModelType.DEEPSEEK:
            print(f"Primary model {model} failed, falling back to DeepSeek")
            fallback_result = await model_factory.generate_content(ModelType.DEEPSEEK, enhanced_prompt, context)
            if fallback_result["success"]:
                return fallback_result["content"]
        
        # 如果所有模型都失败，返回智能错误消息
        # 如果所有模型都失败，返回智能错误消息
        return f"抱歉，AI服务暂时不可用。错误信息：{result.get('error', '未知错误')}。请稍后重试或联系支持团队。"

async def refine_draft(draft: str, feedback: str, model: ModelType, context: Dict[str, Any] = None) -> str:
    """基于反馈优化文书"""
    
    refine_prompt = ENHANCED_PROMPTS["refinement"].format(
        draft=draft,
        feedback=feedback
    )
    
    enhanced_prompt = f"{ENHANCED_PROMPTS['system_base']}\n\n{refine_prompt}"
    
    # 调用模型工厂
    result = await model_factory.generate_content(model, enhanced_prompt, context)
    
    if result["success"]:
        # 记录使用情况
        session_id = context.get("session_id", "unknown") if context else "unknown"
        await cost_tracker.track_usage(
            session_id=session_id,
            model=result["model"],
            tokens=result["tokens_used"],
            cost=result["cost"]
        )
        return result["content"]
    else:
        # 如果优化失败，返回原文书
        print(f"Draft refinement failed: {result.get('error')}")
        return draft

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    try:
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 移除敏感信息，只返回必要的状态信息
        safe_data = {
            "session_id": session_data["session_id"],
            "current_step": session_data["current_step"],
            "major": session_data.get("major"),
            "school": session_data.get("school"),
            "created_at": session_data["created_at"],
            "has_outline": "outline" in session_data,
            "has_draft": "draft" in session_data,
            "has_final": "final_draft" in session_data,
            "completed_at": session_data.get("completed_at")
        }
        
        return safe_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    try:
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        await session_manager.delete_session(session_id)
        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/export")
async def export_session_data(session_id: str):
    """导出会话数据（用于下载文书）"""
    try:
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        export_data = {
            "basic_info": {
                "major": session_data.get("major"),
                "school": session_data.get("school"),
                "created_at": session_data["created_at"]
            },
            "outline": session_data.get("outline"),
            "draft": session_data.get("draft"),
            "final_draft": session_data.get("final_draft"),
            "completion_status": session_data.get("current_step") == "complete"
        }
        
        return export_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/status")
async def get_models_status():
    """获取AI模型状态"""
    try:
        available_models = model_factory.get_available_models()
        budget_status = await cost_tracker.check_budget_limit()
        
        model_status = {}
        for model_type in [ModelType.OPENAI_GPT4, ModelType.CLAUDE_SONNET, ModelType.DEEPSEEK, ModelType.MOCK]:
            if model_type in model_factory._clients:
                client = model_factory._clients[model_type]
                model_status[model_type.value] = {
                    "available": getattr(client, 'available', False),
                    "recommended_for": _get_model_recommendations(model_type)
                }
            else:
                model_status[model_type.value] = {
                    "available": model_type == ModelType.MOCK and MOCK_MODE,
                    "recommended_for": _get_model_recommendations(model_type)
                }
        
        return {
            "available_models": [m.value for m in available_models],
            "model_details": model_status,
            "budget_status": budget_status,
            "mock_mode": MOCK_MODE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_model_recommendations(model_type: ModelType) -> List[str]:
    """获取模型推荐使用场景"""
    recommendations = {
        ModelType.OPENAI_GPT4: ["文书创作", "创意写作", "个性化表达"],
        ModelType.CLAUDE_SONNET: ["大纲规划", "结构化思维", "学术写作", "文书修改"],
        ModelType.DEEPSEEK: ["通用对话", "成本效益优化", "基础文本生成"],
        ModelType.MOCK: ["开发测试", "演示模式"]
    }
    return recommendations.get(model_type, [])

@app.get("/api/analytics/usage")
async def get_usage_analytics():
    """获取使用情况分析（管理员接口）"""
    try:
        budget_status = await cost_tracker.check_budget_limit()
        
        # 这里可以添加更多的分析数据
        analytics = {
            "daily_budget_status": budget_status,
            "system_status": {
                "redis_connected": redis_client is not None,
                "mock_mode": MOCK_MODE,
                "available_models": len(model_factory.get_available_models())
            },
            "service_health": "healthy" if budget_status["remaining"] > 0 else "budget_exceeded"
        }
        
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/{session_id}/reset")
async def reset_session_step(session_id: str, target_step: str = "step1"):
    """重置会话到指定步骤"""
    try:
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if target_step not in STEPS:
            raise HTTPException(status_code=400, detail="Invalid target step")
        
        # 保留基本信息，清除后续步骤的数据
        steps_to_clear = []
        step_order = ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8"]
        target_index = step_order.index(target_step)
        
        # 清除目标步骤之后的所有数据
        if target_index >= 4:  # step5 以后的步骤
            if "outline" in session_data:
                del session_data["outline"]
            if "draft" in session_data:
                del session_data["draft"]
            if "final_draft" in session_data:
                del session_data["final_draft"]
            if "outline_feedback" in session_data:
                del session_data["outline_feedback"]
            if "final_feedback" in session_data:
                del session_data["final_feedback"]
        
        session_data["current_step"] = target_step
        await session_manager.save_session(session_id, session_data)
        
        return {
            "session_id": session_id,
            "current_step": target_step,
            "prompt": STEPS[target_step]["prompt"],
            "message": f"Session reset to {target_step}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"Unexpected error: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail="Internal server error occurred"
    )

# 启动时的初始化
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    print("StoryScape Enhanced API Starting...")
    print(f"Mock Mode: {MOCK_MODE}")
    print(f"Storage: {'Redis' if redis_client else 'Memory'}")
    print(f"Available Models: {len(model_factory.get_available_models())}")
    
    # 初始化预算追踪
    budget_status = await cost_tracker.check_budget_limit()
    print(f"Daily Budget: ${budget_status['daily_budget']:.2f}")
    print("StoryScape Enhanced API Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    print("StoryScape Enhanced API Shutting down...")
    # 这里可以添加清理逻辑，比如关闭数据库连接等
    print("Shutdown complete")

# 主程序入口
if __name__ == "__main__":
    import uvicorn
    
    # 开发环境配置
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting StoryScape Enhanced API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        log_level="info"
    )