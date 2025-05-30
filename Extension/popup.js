// StoryScape Chrome Extension
class StoryscapeApp {
  constructor() {
    this.sessionId = "";
    this.currentStep = "";
    this.prompt = "";
    this.generatedContent = "";
    this.loading = false;
    this.isSessionStarted = false;
    this.apiBase = "http://localhost:5000/api";

    this.initElements();
    this.bindEvents();
    this.loadState();
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

    // 按钮
    this.startBtn = document.getElementById("startBtn");
    this.submitBtn = document.getElementById("submitBtn");
    this.downloadBtn = document.getElementById("downloadBtn");
    this.newBtn = document.getElementById("newBtn");
  }

  bindEvents() {
    this.startBtn.addEventListener("click", () => this.startSession());
    this.submitBtn.addEventListener("click", () => this.submitResponse());
    this.downloadBtn.addEventListener("click", () => this.downloadDraft());
    this.newBtn.addEventListener("click", () => this.startNewSession());

    // 回车提交
    this.userResponse.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && e.ctrlKey) {
        this.submitResponse();
      }
    });
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
    this.setLoading(true, "正在启动会话...");

    try {
      const response = await fetch(`${this.apiBase}/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      this.sessionId = data.session_id;
      this.currentStep = data.current_step;
      this.prompt = data.prompt;
      this.isSessionStarted = true;

      this.showWorkspace();
      this.updateUI();
      await this.saveState();

      this.showStatus("会话已启动！开始您的文书创作之旅 🎉", "success");
    } catch (error) {
      console.error("Error starting session:", error);
      this.showStatus(
        `启动失败: ${error.message}。请确保后端服务器运行在 localhost:5000`,
        "error"
      );
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

    this.setLoading(true, "正在处理您的回应...");

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
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();

      this.currentStep = data.current_step;
      this.prompt = data.prompt;

      if (data.generated_content) {
        this.generatedContent = data.generated_content;
      }

      this.userResponse.value = "";
      this.updateUI();
      await this.saveState();

      this.showStatus("回应已处理 ✅", "success");
    } catch (error) {
      console.error("Error processing step:", error);
      this.showStatus(`处理失败: ${error.message}`, "error");
    } finally {
      this.setLoading(false);
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

    // 更新生成内容
    if (this.generatedContent) {
      this.generatedContentDiv.style.display = "block";
      this.generatedText.textContent = this.generatedContent;

      // 设置标题
      if (this.currentStep === "step6") {
        this.generatedTitle.textContent = "📋 大纲草稿";
      } else if (
        this.currentStep === "step8" ||
        this.currentStep === "complete"
      ) {
        this.generatedTitle.textContent = "📄 文书草稿";
      } else {
        this.generatedTitle.textContent = "✨ 生成内容";
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

    const blob = new Blob([this.generatedContent], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `personal_statement_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.showStatus("文书已下载！ 📥", "success");
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

    // 重置UI
    this.workspace.classList.remove("active");
    this.stepIndicator.style.display = "none";
    this.startContainer.style.display = "block";
    this.generatedContentDiv.style.display = "none";
    this.userResponse.value = "";

    this.showStatus("已重置，可以开始新的文书创作", "success");
  }

  setLoading(loading, message = "") {
    this.loading = loading;
    this.startBtn.disabled = loading;
    this.submitBtn.disabled = loading;

    if (loading) {
      this.startBtn.textContent = "⏳ 处理中...";
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

    setTimeout(() => {
      this.status.style.display = "none";
    }, 3000);
  }
}

// 当页面加载完成时初始化应用
document.addEventListener("DOMContentLoaded", () => {
  new StoryscapeApp();
});
