import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ==========================================
# 1. 配置大模型
# ==========================================
DASHSCOPE_API_KEY = "sk- 省略"

llm = ChatOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-max",
    streaming=True
)

app = FastAPI(title="智能问答机器人")


# ==========================================
# 2. 定义数据结构
# ==========================================
class Message(BaseModel):
    role: str  # 'user' 或 'assistant'
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []
    system_prompt: str = "你是一个严谨的智能问答助手。"
    context: Optional[str] = ""


# ==========================================
# 3. 核心接口
# ==========================================
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    # 构造系统提示词
    final_system = request.system_prompt
    if request.context:
        final_system += f"\n\n参考资料：\n{request.context}"

    # 初始化消息列表，先塞入系统设定
    messages = [SystemMessage(content=final_system)]

    # 严格将历史记录转换为 LangChain 对象并塞入列表
    for msg in request.history:
        if msg.role.lower() == 'user':
            messages.append(HumanMessage(content=msg.content))
        elif msg.role.lower() in ['assistant', 'ai']:
            messages.append(AIMessage(content=msg.content))

    # 最后塞入用户当前的提问
    messages.append(HumanMessage(content=request.query))

    # 打印日志到控制台，方便你观察后端到底发了什么给大模型
    print(f"--- 调试信息：本次发送给模型的总消息数: {len(messages)} ---")
    for m in messages:
        print(f"[{type(m).__name__}]: {m.content[:20]}...")

    async def generate_stream():
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
