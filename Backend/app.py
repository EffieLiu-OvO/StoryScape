from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import time
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用于session
CORS(app, supports_credentials=True)

# 模拟缓存
cache = {}

# 环境变量存储API密钥
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-key")

# 存储用户会话状态
sessions = {}

# 预设的文书创作流程步骤
STEPS = {
    "step1": {
        "name": "收集基本信息",
        "prompt": "请提供您申请的专业领域和目标学校。这将帮助我们为您定制文书创作流程。"
    },
    "step2": {
        "name": "材料收集",
        "prompt": "为了创建个性化的文书，请提供以下材料：\n1. 您的教育背景\n2. 相关课程和成绩\n3. 实习/项目经历\n4. 获奖情况\n5. 课外活动\n您可以分条列出，无需格式化。"
    },
    "step3": {
        "name": "强调重点",
        "prompt": "请分享您最想在文书中突出的2-3个经历或特质。这些会成为您文书的核心。"
    },
    "step4": {
        "name": "动机和目标",
        "prompt": "请描述您为什么对所选专业感兴趣，以及您的长期职业目标。这将成为文书的重要部分。"
    },
    "step5": {
        "name": "生成大纲",
        "internal_prompt": "请参考学生的素材，创建一个合理的文书大纲。文书应包含5-6个段落：\n1. 引人入胜的动机段，描述学生如何对该领域产生兴趣\n2. 学术背景段，包含相关课程和研究\n3-5. 2-3个相互关联但内核不同的实践段，展示学生的经历和成长\n6. Why School段和未来规划\n\n请从学生的经历中识别一个特定领域，将所有经历串联成一个成长故事。避免AI痕迹，保持人情味。提供清晰的段落主题和关键点。"
    },
    "step6": {
        "name": "完善大纲",
        "prompt": "请查看生成的大纲，并提供反馈。有哪些部分您希望调整或强调？"
    },
    "step7": {
        "name": "生成文书",
        "internal_prompt": "基于批准的大纲和学生的所有材料，创建一篇完整的个人陈述。确保文书：\n1. 语调自然，避免AI风格的写作\n2. 故事线流畅，各部分衔接自然\n3. 展现学生的个性和成长\n4. 符合学术申请的专业标准\n5. 强调为什么学生适合该项目\n\n保持真实性，使用具体细节和个人反思，避免过度修饰。"
    },
    "step8": {
        "name": "最终审核",
        "prompt": "请查看生成的文书草稿，并提供反馈。有哪些部分需要进一步修改？"
    }
}

@app.route('/api/start', methods=['POST'])
def start_session():
    data = request.json
    session_id = str(int(time.time()))  # 简单生成会话ID
    
    sessions[session_id] = {
        "current_step": "step1",
        "major": "",
        "school": "",
        "materials": {},
        "outline": "",
        "draft": ""
    }
    
    return jsonify({
        "session_id": session_id,
        "current_step": "step1",
        "prompt": STEPS["step1"]["prompt"]
    })

@app.route('/api/process', methods=['POST'])
def process_step():
    data = request.json
    session_id = data.get('session_id')
    user_response = data.get('response', '')
    
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    session_data = sessions[session_id]
    current_step = session_data["current_step"]
    
    # 处理当前步骤的响应
    if current_step == "step1":
        # 解析专业和学校
        parts = user_response.split(',')
        if len(parts) >= 2:
            session_data["major"] = parts[0].strip()
            session_data["school"] = parts[1].strip()
        else:
            session_data["major"] = user_response
        
        next_step = "step2"
    
    elif current_step == "step2":
        # 存储收集的材料
        session_data["materials"]["background"] = user_response
        next_step = "step3"
    
    elif current_step == "step3":
        session_data["materials"]["highlights"] = user_response
        next_step = "step4"
    
    elif current_step == "step4":
        session_data["materials"]["motivation"] = user_response
        next_step = "step5"
        
        # 在步骤5中，我们生成大纲
        # 这里应该调用实际的AI API，但现在我们简化它
        materials = json.dumps(session_data["materials"])
        major = session_data["major"]
        school = session_data["school"]
        
        # 使用内部prompt生成大纲
        internal_prompt = STEPS["step5"]["internal_prompt"]
        full_prompt = f"{internal_prompt}\n\n专业: {major}\n学校: {school}\n\n学生材料:\n{materials}"
        
        # 模拟AI调用
        outline = generate_outline(full_prompt, major)
        session_data["outline"] = outline
        
        # 提供下一步的提示，但包含生成的大纲
        return jsonify({
            "session_id": session_id,
            "current_step": "step6",
            "prompt": STEPS["step6"]["prompt"],
            "generated_content": outline
        })
    
    elif current_step == "step6":
        # 存储大纲反馈
        session_data["outline_feedback"] = user_response
        next_step = "step7"
        
        # 生成文书草稿
        internal_prompt = STEPS["step7"]["internal_prompt"]
        outline = session_data["outline"]
        feedback = user_response
        materials = json.dumps(session_data["materials"])
        
        full_prompt = f"{internal_prompt}\n\n大纲: {outline}\n\n大纲反馈: {feedback}\n\n学生材料: {materials}"
        
        # 模拟AI调用生成文书
        draft = generate_draft(full_prompt, session_data["major"])
        session_data["draft"] = draft
        
        return jsonify({
            "session_id": session_id,
            "current_step": "step8",
            "prompt": STEPS["step8"]["prompt"],
            "generated_content": draft
        })
    
    elif current_step == "step8":
        # 最终文书反馈
        session_data["draft_feedback"] = user_response
        
        # 根据反馈修改文书
        draft = session_data["draft"]
        feedback = user_response
        
        final_draft = revise_draft(draft, feedback)
        session_data["final_draft"] = final_draft
        
        return jsonify({
            "session_id": session_id,
            "current_step": "complete",
            "prompt": "您的文书已完成! 您可以下载最终版本或继续修改。",
            "generated_content": final_draft
        })
    
    else:
        return jsonify({"error": "Invalid step"}), 400
    
    # 更新会话中的当前步骤
    session_data["current_step"] = next_step
    
    # 返回下一步的提示
    return jsonify({
        "session_id": session_id,
        "current_step": next_step,
        "prompt": STEPS[next_step]["prompt"]
    })

def generate_outline(prompt, major):
    # 真实项目中，这里应该调用OpenAI或其他AI API
    # 为演示目的，返回模拟的大纲
    
    # 模拟不同专业的个性化大纲
    if "城市管理" in major or "城市规划" in major:
        return """
# 城市管理个人陈述大纲

## 第一段：引言/动机
- 通过亲身参与社区更新项目，发现城市空间如何影响居民生活质量
- 描述一个具体转折点：参与某个社区项目时的启发性时刻
- 引出对城市可持续发展和宜居性的关注

## 第二段：学术背景
- 相关课程：城市规划理论、公共政策分析、可持续发展研究
- 研究项目：分析城市边缘社区的公共空间使用模式
- 理论基础如何塑造了对城市问题的理解

## 第三段：实践经历一 - 政府视角
- 在地方政府实习期间接触的城市管理实践
- 参与了哪些规划或政策制定过程
- 学习到的关于多方利益平衡的经验

## 第四段：实践经历二 - 社区视角
- 与社区组织合作的项目经历
- 如何促进居民参与城市空间规划
- 从基层视角理解城市问题的价值

## 第五段：实践经历三 - 创新解决方案
- 运用技术或创新方法解决城市问题的尝试
- 具体项目及其影响
- 如何整合前两段经历中获得的见解

## 第六段：Why School & 未来规划
- 为什么选择港大的城市管理项目
- 项目特色如何与您的兴趣和经验匹配
- 毕业后如何将所学应用于改善城市生活质量的长期职业目标

**核心主题**：将您的经历串联为"从多角度理解和改善城市空间与社区生活质量"的旅程，展示您如何逐步形成了全面、深入的城市管理视角。
        """
    elif "计算机" in major or "软件" in major or "computer" in major.lower():
        return """
# 计算机科学个人陈述大纲

## 第一段：引言/动机
- 描述首次接触编程的启发性经历
- 通过某个特定项目发现技术解决实际问题的潜力
- 引出对专业学习的热忱

## 第二段：学术背景
- 相关课程：算法设计、数据结构、人工智能等
- 研究项目：某个引人入胜的计算机科学问题
- 理论知识如何支持您的实践能力

## 第三段至第五段：项目经历
- 三个代表性项目，展示不同技能和成长
- 每个项目面临的挑战和解决方案
- 技术选择和设计决策的思考过程

## 第六段：Why School & 未来规划
- 目标学校的计算机科学项目特色与您的匹配点
- 毕业后的研究或行业发展计划
- 长期职业愿景
        """
    else:
        return """
# 个人陈述大纲

## 第一段：引言/动机
- 描述您对所选领域产生兴趣的关键经历
- 引出您希望在该领域深造的动机

## 第二段：学术背景
- 相关课程和研究经历
- 如何为进一步学习奠定基础

## 第三段：实践经历一
- 第一个关键项目或实习
- 所获技能和见解

## 第四段：实践经历二
- 第二个关键项目或实习
- 如何深化和拓展您的专业视角

## 第五段：实践经历三
- 第三个关键项目或实习
- 如何整合前两段经历的经验和教训

## 第六段：Why School & 未来规划
- 为什么选择该学校的该项目
- 您的长期职业目标和规划
        """

def generate_draft(prompt, major):
    # 模拟生成文书草稿
    # 真实项目中，这里应调用AI API
    
    if "城市管理" in major or "城市规划" in major:
        return """
# 城市空间与社区活力：我的城市管理之旅

在大学二年级的一个周末，我参与了家乡一个老旧社区的改造志愿项目。当我看到一个简单的口袋公园如何转变了居民的互动方式，我感到震撼。一位年长居民对我说："这不只是一块绿地，这是我们社区的心脏。"这一刻彻底改变了我对城市空间的理解——它们不仅是物理环境，更是塑造社区活力和居民生活质量的关键。这种认识引导我踏上了城市管理的学习之路，希望能够为创造更宜居、更可持续的城市环境贡献自己的力量。

我的学术背景为我提供了坚实的理论基础。通过学习城市规划理论、公共政策分析和可持续发展研究等核心课程，我开始理解城市系统的复杂性。特别是在"城市更新与社会公平"课程的研究项目中，我分析了三个不同城市边缘社区的公共空间使用模式，发现了空间设计与社区凝聚力之间的紧密关系。这一研究不仅让我获得了数据收集与分析的实践经验，更重要的是深化了我对城市规划必须平衡效率、公平与可持续性的理解。

我在市政府城市规划部门的实习提供了宝贵的政府视角。在那里，我协助评估了一项城市更新计划对当地商业环境的潜在影响。最令我印象深刻的是决策过程中的复杂性——如何平衡发展需求、财政限制和居民意见。我参与的公众咨询会议让我亲眼目睹了政策制定者如何在各种压力下工作，也让我认识到有效沟通和利益平衡的重要性。这段经历教会了我从宏观角度思考城市问题，考虑政策的长远影响而非短期效果。

然而，政府视角只是拼图的一部分。随后我加入了一个社区发展非营利组织，这让我获得了完全不同的基层视角。在那里，我协调了一个参与式规划项目，帮助当地居民表达他们对社区公共空间的需求和愿景。我们组织了工作坊，使用创新方法如社区地图绘制和模型构建，让不同年龄和背景的居民都能参与进来。这一过程充满挑战——要处理各种冲突意见，确保弱势群体的声音被听到，同时保持项目的可行性。但当我看到最终方案如何真实反映了社区需求，我深刻体会到了自下而上规划的价值。

将这些经验结合起来，我开始探索如何利用技术创新解决城市问题。在一个智慧城市创新比赛中，我带领团队开发了一个移动应用原型，旨在改善老年人对公共交通的使用体验。我们结合了在政府实习期间对系统限制的理解，以及在社区工作中获得的用户需求洞察。虽然这个项目规模不大，但它代表了我对城市管理的愿景——利用数据和技术，但始终以人为本，关注那些最容易被忽视的群体。我们的方案获得了比赛二等奖，更重要的是，这证明了我能够整合不同视角来创造实用解决方案。

香港大学的城市管理项目正是我理想的下一步。贵校在亚洲城市研究领域的领先地位，特别是在城市更新与社会包容性方面的专长，与我的研究兴趣高度契合。我特别欣赏该项目强调跨学科方法和案例研究的教学理念，这将帮助我更全面地理解城市问题。完成学业后，我计划回到国内，在地方政府或规划咨询机构工作，将所学知识应用于正在快速发展的中国城市。长期来看，我希望能够专注于城市公共空间的包容性设计，确保城市发展能够惠及所有居民，特别是弱势群体。

我的城市管理之旅始于一个社区公园，经历了政府、社区和创新实践的多重视角。这些经历塑造了我对城市空间的理解——它们是物理环境，也是社会结构，更是连接人与人的纽带。在香港大学的学习将是这段旅程的重要里程碑，帮助我实现为城市创造更美好未来的愿望。
        """
    else:
        return """
# 个人陈述草稿

[这里是根据大纲生成的完整个人陈述草稿，针对学生的专业和经历定制]

第一段：引言/动机
...

第二段：学术背景
...

第三至五段：实践经历
...

第六段：Why School & 未来规划
...
        """

def revise_draft(draft, feedback):
    # 模拟根据反馈修改文书
    # 真实项目中应调用AI API
    
    # 简单演示，假设只是在文书末尾添加一个修改说明
    return draft + f"\n\n## 根据反馈修改的说明\n\n{feedback}\n\n[这里是根据以上反馈修改后的最终文书]"

if __name__ == '__main__':
    app.run(debug=True, port=5000)