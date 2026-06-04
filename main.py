import os
import json
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse

# 向量库与切片相关组件
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 本地多格式文件解析库
import pdfplumber
from docx import Document as DocxDocument
import openpyxl

import dashscope
from dashscope import MultiModalEmbedding, Generation

# 基础配置
DASHSCOPE_API_KEY = "sk-密钥"
os.makedirs("vector_store", exist_ok=True)

dashscope.api_key = DASHSCOPE_API_KEY


# 自定义 Embedding 类（符合 LangChain 规范的对象）
class DashScopeEmbeddings:
    def __init__(self, model: str = "qwen3-vl-embedding"):
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        try:
            resp = MultiModalEmbedding.call(
                model=self.model,
                input=[{'text': text}]
            )
            if resp.status_code == 200:
                return resp.output['embeddings'][0]['embedding']
            else:
                raise Exception(f"Embedding API 失败: {resp.message}")
        except Exception as e:
            print(f"Embedding 错误: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


# 初始化实例
embedding = DashScopeEmbeddings(model="qwen3-vl-embedding")

app = FastAPI(title="智能问答机器人后端API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []
    kb_id: Optional[str] = "default"
    model_name: Optional[str] = "qwen-max"


# 文档解析逻辑
def extract_text_from_file(file_path: str, ext: str) -> str:
    text = ""
    ext = ext.lower()
    try:
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: text += page_text + "\n"
        elif ext == ".docx":
            doc = DocxDocument(file_path)
            for para in doc.paragraphs:
                if para.text: text += para.text + "\n"
        elif ext == ".xlsx":
            wb = openpyxl.load_workbook(file_path)
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = " ".join([str(cell) for cell in row if cell is not None])
                    if row_text.strip(): text += row_text + "\n"
    except Exception as e:
        print(f"解析文件错误: {e}")
    return text.strip()


def create_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# 向量检索逻辑
def retrieve_context(query: str, kb_id: str) -> str:
    try:
        db_path = f"vector_store/{kb_id}"
        if not os.path.exists(f"{db_path}/index.faiss"):
            return ""
        vectorstore = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"检索异常: {e}")
        return ""


# ==========================================
# API 核心接口
# ==========================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), kb_id: str = Form("default")):
    tmp_path = None
    try:
        ext = os.path.splitext(file.filename)[-1].lower()
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        text = extract_text_from_file(tmp_path, ext)
        chunks = create_chunks(text)
        documents = [Document(page_content=c, metadata={"source": file.filename}) for c in chunks]

        db_path = f"vector_store/{kb_id}"
        if os.path.exists(f"{db_path}/index.faiss"):
            vectorstore = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
            vectorstore.add_documents(documents)
        else:
            vectorstore = FAISS.from_documents(documents, embedding)

        vectorstore.save_local(db_path)
        return {"status": "ok", "msg": "上传成功"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        try:
            # 1. 模型纠错：防止多模态模型进入纯文本接口
            current_model = request.model_name
            if "-vl-" in current_model.lower():
                print(f"⚠️ 自动修复：将多模态模型 {current_model} 切换为文本模型 qwen-plus")
                current_model = "qwen-plus"

            # 2. 检索上下文
            context = retrieve_context(request.query, request.kb_id)
            system_prompt = f"资料：\n{context}\n回答问题：" if context else "你是一个助手。"

            messages = [{"role": "system", "content": system_prompt}]
            for m in request.history:
                messages.append({"role": m.role, "content": m.content})
            messages.append({"role": "user", "content": request.query})

            print(f"🚀 正在调度模型: {current_model}")

            responses = Generation.call(
                model=current_model,
                messages=messages,
                result_format='message',
                stream=True,
                incremental_output=True
            )

            has_thought_opened = False

            for response in responses:
                if response.status_code == 200:
                    choice = response.output.choices[0]
                    message = getattr(choice, 'message', {})

                    # 🟢 最高防御等级提取内容
                    reasoning = ""
                    content = ""

                    # 提取思考内容（捕获所有潜在的 Key/Attribute 错误）
                    try:
                        reasoning = getattr(message, 'reasoning_content', '')
                    except:
                        try:
                            reasoning = message.get('reasoning_content', '')
                        except:
                            reasoning = ""

                    # 提取正文内容
                    try:
                        content = getattr(message, 'content', '')
                    except:
                        try:
                            content = message.get('content', '')
                        except:
                            content = ""

                    # 发送思考流
                    if reasoning:
                        if not has_thought_opened:
                            yield "data: <think>\n\n"
                            has_thought_opened = True
                        yield f"data: {reasoning}\n\n"

                    # 发送正文流
                    if content:
                        if has_thought_opened:
                            yield "data: </think>\n\n\n"
                            has_thought_opened = False
                        yield f"data: {content}\n\n"
                else:
                    yield f"data: 错误({response.code}): {response.message}\n\n"
                    break

            yield "data: [DONE]\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: 错误: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/kb/list")
async def list_kb():
    kbs = []
    if os.path.exists("vector_store"):
        for d in os.listdir("vector_store"):
            if os.path.isdir(os.path.join("vector_store", d)):
                kbs.append(d)
    return {"knowledge_bases": kbs}


@app.get("/")
async def root():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
