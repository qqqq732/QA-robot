import os
import json
import tempfile
import traceback
import shutil
import hashlib
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

# 导入百炼原生 SDK 组件
import dashscope
from dashscope import MultiModalEmbedding, Generation

# 基础配置
DASHSCOPE_API_KEY = "sk-密钥"
os.makedirs("vector_store", exist_ok=True)
dashscope.api_key = DASHSCOPE_API_KEY


class DashScopeEmbeddings:
    def __init__(self, model: str = "qwen3-vl-embedding"):
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        resp = MultiModalEmbedding.call(model=self.model, input=[{'text': text}])
        if resp.status_code == 200: return resp.output['embeddings'][0]['embedding']
        raise Exception(f"Embedding API 失败: {resp.message}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


embedding = DashScopeEmbeddings(model="qwen3-vl-embedding")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []  # noqa
    kb_id: Optional[str] = ""
    model_name: Optional[str] = "qwen-max"


def get_safe_id(filename: str) -> str:
    return hashlib.md5(filename.encode('utf-8')).hexdigest()


def extract_text_from_file(file_path: str, ext: str) -> str:
    text = ""
    try:
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t: text += t + "\n"
        elif ext == ".docx":
            doc = DocxDocument(file_path)
            for para in doc.paragraphs: text += para.text + "\n"
        elif ext == ".xlsx":
            wb = openpyxl.load_workbook(file_path)
            for s in wb.worksheets:
                for row in s.iter_rows(values_only=True):
                    text += " ".join([str(c) for c in row if c is not None]) + "\n"
    except Exception as e:
        print(f"解析错误: {e}")
    return text.strip()


# API 核心接口

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    tmp_path = None
    try:
        filename = file.filename
        ext = os.path.splitext(filename)[-1].lower()
        content = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        text = extract_text_from_file(tmp_path, ext)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        if not chunks: return {"status": "error", "msg": "文档无有效文本"}

        safe_id = get_safe_id(filename)
        db_path = os.path.join("vector_store", safe_id)
        os.makedirs(db_path, exist_ok=True)

        with open(os.path.join(db_path, "original_name.txt"), "w", encoding="utf-8") as f:
            f.write(filename)

        documents = [Document(page_content=c, metadata={"source": filename}) for c in chunks]

        if os.path.exists(os.path.join(db_path, "index.faiss")):
            vs = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
            vs.add_documents(documents)
        else:
            vs = FAISS.from_documents(documents, embedding)

        vs.save_local(db_path)
        return {"status": "ok", "kb_id": safe_id, "kb_name": filename}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "msg": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        try:
            context_list = []
            if os.path.exists("vector_store"):
                for safe_id in os.listdir("vector_store"):
                    db_path = os.path.join("vector_store", safe_id)
                    if os.path.isdir(db_path) and os.path.exists(os.path.join(db_path, "index.faiss")):
                        try:
                            vs = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
                            docs = vs.similarity_search(request.query, k=2)  # 每个文档找最相似的2个片段
                            for d in docs:
                                context_list.append(d.page_content)
                        except Exception as e:
                            print(f"联合检索单库失败({safe_id}): {e}")

            context = "\n\n".join(context_list)

            system_prompt = (
                f"你是一个智能助手。请基于以下参考资料回答问题。\n"
                f"【极其重要】：如果答案包含多个要点，请务必使用‘1.’、‘2.’、‘3.’等序号进行‘分段、分点’回答。每个要点独立成段，多使用换行符分隔开，严禁把所有内容挤在单一长段落里面！\n\n"
                f"参考资料：\n{context}\n\n"
                f"当前问题：{request.query}"
            ) if context else (
                "你是一个助手。如果回答内容较长，请务必使用换行符（Enter）分成多段，并使用1. 2. 3.分点列出，不要挤在一段里。"
            )

            api_messages = [{"role": "system", "content": system_prompt}]
            for m in request.history:
                api_messages.append({"role": m.role, "content": m.content})
            api_messages.append({"role": "user", "content": request.query})

            chosen_model = request.model_name
            if "-vl-" in chosen_model.lower():
                chosen_model = "qwen-plus"

            responses = Generation.call(
                model=chosen_model,
                messages=api_messages,
                result_format='message',
                stream=True,
                incremental_output=True
            )

            has_thought = False
            for response in responses:
                if response.status_code == 200:
                    choice = response.output.choices[0]
                    message = getattr(choice, 'message', {})

                    reasoning = ""
                    content = ""

                    try:
                        reasoning = getattr(message, 'reasoning_content', '')
                    except:
                        try:
                            reasoning = message.get('reasoning_content', '')
                        except:
                            reasoning = ""

                    try:
                        content = getattr(message, 'content', '')
                    except:
                        try:
                            content = message.get('content', '')
                        except:
                            content = ""

                    if reasoning:
                        if not has_thought:
                            yield "data: <think>\n\n"
                            has_thought = True
                        yield f"data: {reasoning}\n\n"
                    if content:
                        if has_thought:
                            yield "data: </think>\n\n\n"
                            has_thought = False
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
        for safe_id in os.listdir("vector_store"):
            p = os.path.join("vector_store", safe_id)
            name_file = os.path.join(p, "original_name.txt")
            if os.path.isdir(p) and os.path.exists(name_file):
                with open(name_file, "r", encoding="utf-8") as f:
                    kbs.append({"id": safe_id, "name": f.read().strip()})
    return {"knowledge_bases": kbs}


@app.post("/api/kb/delete")
async def delete_kb(data: dict):
    kb_id = data.get("kb_id")
    shutil.rmtree(os.path.join("vector_store", kb_id), ignore_errors=True)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
