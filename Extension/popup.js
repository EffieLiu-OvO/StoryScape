// Extension/popup.js - Updated version
class StoryscapeApp {
  constructor() {
    this.sessionId = "";
    this.currentStep = "";
    this.prompt = "";
    this.generatedContent = "";
    this.loading = false;
    this.isSessionStarted = false;
    this.apiBase = "http://localhost:8000/api";
    this.costInfo = null;
    this.modelUsed = null;

    this.initElements();
    this.bindEvents();
    this.loadState();
    this.checkAPIHealth();
  }

  initElements() {
    this.startContainer = document.getElementById("startContainer");
    this.workspace = document.getElementById("workspace");
    this.stepIndicator = document.getElementById("stepIndicator");
    this.promptContent = document.getElementById("promptContent");
    this.generatedContentDiv = document.getElementById("generatedContent");
    this.generatedTitle = document.getElementById("generatedTitle");
    this.generatedText = document.getElementById("generatedText");
    this.responseArea = document.getElementById("responseArea");
    this.userResponse = document.getElementById("userResponse");
    this.completeActions = document.getElementById("completeActions");
    this.status = document.getElementById("status");
    this.costDisplay = document.getElementById("costDisplay");
    this.modelInfo = document.getElementById("modelInfo");

    // 按钮
    this.startBtn = document.getElementById("startBtn");
    this.submitBtn = document.getElementById("submitBtn");
    this.downloadBtn = document.getElementById("downloadBtn");
    this.newBtn = document.getElementById("newBtn");
    this.costBtn = document.getElementById("costBtn");
  }

  bindEvents() {
    this.startBtn.addEventListener("click", () => this.startSession());
    this.submitBtn.addEventListener("click", () => this.submitResponse());
    this.downloadBtn.addEventListener("click", () => this.downloadDraft());
    this.newBtn.addEventListener("click", () => this.startNewSession());
    this.costBtn.addEventListener("click", () => this.showCostBreakdown());

    // 回车提交
    this.userResponse.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && e.ctrlKey) {
        this.submitResponse();
      }
    });

    // 实时字数统计
    this.userResponse.addEventListener("input", () => {
      this.updateWordCount();
    });
  }

  async checkAPIHealth() {
    try {
      const response = await fetch(
        `${this.apiBase.replace("/api", "")}/health`
      );
      if (response.ok) {
        const health = await response.json();
        this.showStatus(`✅ API连接正常 (Redis: ${health.redis})`, "success");
      } else {
        throw new Error("Health check failed");
      }
    } catch (error) {
      this.showStatus(
        "⚠️ 无法连接到后端服务，请确保FastAPI服务器运行在localhost:8000",
        "error"
      );
    }
  }

  async loadState() {
    try {
      const result = await chrome.storage.local.get(["storyscapeSession"]);
      const saved = result.storyscapeSession;

      if (saved && saved.sessionId) {
        this.sessionId = saved.sessionId;
        this.currentStep = saved.currentStep;
        this.prompt = saved.prompt;
        this.generatedContent = saved.generatedContent;
        this.isSessionStarted = true;

        this.showWorkspace();
        this.updateUI();

        // 加载成本信息
        await this.loadCostInfo();
      }
    } catch (error) {
      console.error("Failed to load state:", error);
    }
  }

  async saveState() {
    try {
      await chrome.storage.local.set({
        storyscapeSession: {
          sessionId: this.sessionId,
          currentStep: this.currentStep,
          prompt: this.prompt,
          generatedContent: this.generatedContent,
        },
      });
    } catch (error) {
      console.error("Failed to save state:", error);
    }
  }

  async startSession() {
    this.setLoading(true, "正在启动AI文书助手...");

    try {
      const response = await fetch(`${this.apiBase}/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: "chrome_extension_user",
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      this.sessionId = data.session_id;
      this.currentStep = data.current_step;
      this.prompt = data.prompt;
      this.modelUsed = data.model_used;
      this.isSessionStarted = true;

      this.showWorkspace();
      this.updateUI();
      await this.saveState();

      this.showStatus("🎉 文书创作会话已启动！", "success");
    } catch (error) {
      console.error("Error starting session:", error);
      this.showStatus(`启动失败: ${error.message}`, "error");
    } finally {
      this.setLoading(false);
    }
  }

  async submitResponse() {
    const response = this.userResponse.value.trim();
    if (!response) {
      this.showStatus("请输入您的回应", "error");
      return;
    }

    this.setLoading(true, "AI正在分析您的回应...");

    try {
      const res = await fetch(`${this.apiBase}/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: this.sessionId,
          response: response,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => null);
        throw new Error(errorData?.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();

      this.currentStep = data.current_step;
      this.prompt = data.prompt;
      this.modelUsed = data.model_used;

      if (data.generated_content) {
        this.generatedContent = data.generated_content;
      }

      // 显示处理时间和模型信息
      if (data.processing_time) {
        this.showStatus(
          `✅ 处理完成 (${data.processing_time.toFixed(2)}s, 使用${
            data.model_used
          })`,
          "success"
        );
      }

      this.userResponse.value = "";
      this.updateUI();
      await this.saveState();
      await this.loadCostInfo(); // 更新成本信息
    } catch (error) {
      console.error("Error processing step:", error);
      this.showStatus(`处理失败: ${error.message}`, "error");
    } finally {
      this.setLoading(false);
    }
  }

  async loadCostInfo() {
    if (!this.sessionId) return;

    try {
      const response = await fetch(
        `${this.apiBase}/session/${this.sessionId}/cost`
      );
      if (response.ok) {
        this.costInfo = await response.json();
        this.updateCostDisplay();
      }
    } catch (error) {
      console.log("Could not load cost info:", error);
    }
  }

  updateCostDisplay() {
    if (!this.costInfo || !this.costDisplay) return;

    const cost = this.costInfo.cost_summary;
    this.costDisplay.innerHTML = `
      <div class="cost-summary">
        💰 本次创作成本: $${cost.estimated_cost?.toFixed(4) || "0.0000"}
        <br>🔢 总token使用: ${cost.total_tokens || 0}
      </div>
    `;
  }

  async showCostBreakdown() {
    if (!this.costInfo) {
      this.showStatus("暂无成本信息", "info");
      return;
    }

    const breakdown = this.costInfo.model_breakdown;
    let message = "📊 模型使用详情:\n\n";

    for (const [model, info] of Object.entries(breakdown)) {
      message += `${model}:\n`;
      message += `  - 调用次数: ${info.calls}\n`;
      message += `  - Token数: ${info.tokens}\n`;
      message += `  - 费用: $${info.cost.toFixed(4)}\n\n`;
    }

    // 创建模态框显示详细信息
    this.showDetailedInfo("成本分析", message);
  }

  showDetailedInfo(title, content) {
    // 简单的alert显示，实际项目中可以用更好的模态框
    alert(`${title}\n\n${content}`);
  }

  updateWordCount() {
    const text = this.userResponse.value;
    const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
    const charCount = text.length;

    // 更新字数显示（如果有对应元素）
    const wordCountDisplay = document.getElementById("wordCount");
    if (wordCountDisplay) {
      wordCountDisplay.textContent = `${wordCount} 词 / ${charCount} 字符`;
    }
  }

  showWorkspace() {
    this.startContainer.style.display = "none";
    this.workspace.classList.add("active");
    this.stepIndicator.style.display = "flex";
  }

  updateUI() {
    // 更新步骤指示器
    this.updateStepIndicator();

    // 更新提示内容
    this.promptContent.textContent = this.prompt;

    // 更新模型信息显示
    if (this.modelUsed && this.modelInfo) {
      this.modelInfo.textContent = `当前使用: ${this.modelUsed}`;
    }

    // 更新生成内容
    if (this.generatedContent) {
      this.generatedContentDiv.style.display = "block";
      this.generatedText.textContent = this.generatedContent;

      // 设置标题
      if (this.currentStep === "step6") {
        this.generatedTitle.textContent = "📋 个性化大纲";
      } else if (
        this.currentStep === "step8" ||
        this.currentStep === "complete"
      ) {
        this.generatedTitle.textContent = "📄 个人陈述草稿";
      } else {
        this.generatedTitle.textContent = "✨ AI生成内容";
      }
    } else {
      this.generatedContentDiv.style.display = "none";
    }

    // 控制界面显示
    if (this.currentStep === "complete") {
      this.responseArea.style.display = "none";
      this.completeActions.style.display = "flex";
    } else {
      this.responseArea.style.display = "block";
      this.completeActions.style.display = "none";
    }

    // 更新成本显示
    this.updateCostDisplay();
  }

  updateStepIndicator() {
    const steps = [
      "step1",
      "step2",
      "step3",
      "step4",
      "step5",
      "step6",
      "step7",
      "step8",
    ];
    const currentIndex = steps.indexOf(this.currentStep);

    document.querySelectorAll(".step").forEach((step, index) => {
      step.classList.remove("active", "completed");

      if (index === currentIndex) {
        step.classList.add("active");
      } else if (index < currentIndex) {
        step.classList.add("completed");
      }
    });
  }

  downloadDraft() {
    if (!this.generatedContent) {
      this.showStatus("没有可下载的内容", "error");
      return;
    }

    // 添加元数据到下载内容
    const metadata = `\n\n--- 创作信息 ---\n会话ID: ${
      this.sessionId
    }\n创作时间: ${new Date().toLocaleString()}\n使用模型: ${
      this.modelUsed || "多模型协作"
    }`;

    const fullContent = this.generatedContent + metadata;

    const blob = new Blob([fullContent], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `storyscape_statement_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.showStatus("📥 个人陈述已下载！", "success");
  }

  async startNewSession() {
    // 清除存储的状态
    await chrome.storage.local.remove(["storyscapeSession"]);

    // 重置所有状态
    this.sessionId = "";
    this.currentStep = "";
    this.prompt = "";
    this.generatedContent = "";
    this.isSessionStarted = false;
    this.costInfo = null;
    this.modelUsed = null;

    // 重置UI
    this.workspace.classList.remove("active");
    this.stepIndicator.style.display = "none";
    this.startContainer.style.display = "block";
    this.generatedContentDiv.style.display = "none";
    this.userResponse.value = "";

    if (this.costDisplay) {
      this.costDisplay.innerHTML = "";
    }

    this.showStatus("已重置，可以开始新的文书创作", "success");
  }

  setLoading(loading, message = "") {
    this.loading = loading;
    this.startBtn.disabled = loading;
    this.submitBtn.disabled = loading;

    if (loading) {
      this.startBtn.textContent = "⏳ 启动中...";
      this.submitBtn.textContent = "⏳ 处理中...";
      if (message) {
        this.showStatus(message, "loading");
      }
    } else {
      this.startBtn.textContent = "🚀 开始创作";
      this.submitBtn.textContent = "📤 提交";
    }
  }

  showStatus(message, type = "info") {
    this.status.textContent = message;
    this.status.className = `status ${type}`;
    this.status.style.display = "block";

    // 自动隐藏（除了错误信息）
    if (type !== "error") {
      setTimeout(() => {
        this.status.style.display = "none";
      }, 3000);
    }
  }
}

// 当页面加载完成时初始化应用
document.addEventListener("DOMContentLoaded", () => {
  new StoryscapeApp();
});
