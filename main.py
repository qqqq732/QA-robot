import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse

# 向量库相关
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 本地文件解析
import pdfplumber
from docx import Document as DocxDocument
import openpyxl

# 导入 dashscope
import dashscope
from dashscope import MultiModalEmbedding

# ==========================================
# 配置
# ==========================================
DASHSCOPE_API_KEY = "sk-9bf45d009d1c487d857aaf54e3a2bbe4"
os.makedirs("vector_store", exist_ok=True)

# 设置 dashscope API key
dashscope.api_key = DASHSCOPE_API_KEY

# ==========================================
# 自定义 Embedding 类（使用 qwen3-vl-embedding）
# ==========================================
class DashScopeEmbeddings:
    """使用 dashscope 的 qwen3-vl-embedding 模型"""
    
    def __init__(self, model: str = "qwen3-vl-embedding"):
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """生成单个文本的 embedding"""
        try:
            # 调用多模态 embedding API
            resp = MultiModalEmbedding.call(
                model=self.model,
                input=[{'text': text}]
            )
            
            if resp.status_code == 200:
                return resp.output['embeddings'][0]['embedding']
            else:
                raise Exception(f"Embedding API 调用失败: {resp.message}")
        except Exception as e:
            print(f"Embedding 错误: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成 embedding"""
        embeddings = []
        for i, text in enumerate(texts):
            print(f"生成 embedding {i+1}/{len(texts)}...")
            embeddings.append(self.embed_query(text))
        return embeddings

# ==========================================
# LLM 配置（使用 dashscope 兼容模式）
# ==========================================
llm = ChatOpenAI(
    model="qwen-max",
    openai_api_key=DASHSCOPE_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=True,
    temperature=0.7
)

# 创建 embedding 实例
embedding = DashScopeEmbeddings(model="qwen3-vl-embedding")

app = FastAPI(title="文档问答机器人")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 数据结构
# ==========================================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []
    kb_id: Optional[str] = "default"

# ==========================================
# 简化的文件解析
# ==========================================
def extract_text_from_file(file_path: str, ext: str) -> str:
    """提取文件文本内容"""
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
                    if page_text:
                        text += page_text + "\n"
        
        elif ext == ".docx":
            doc = DocxDocument(file_path)
            for para in doc.paragraphs:
                if para.text:
                    text += para.text + "\n"
        
        elif ext == ".xlsx":
            wb = openpyxl.load_workbook(file_path)
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = " ".join([str(cell) for cell in row if cell is not None])
                    if row_text.strip():
                        text += row_text + "\n"
    
    except Exception as e:
        print(f"解析错误: {e}")
        return ""
    
    return text.strip()

# ==========================================
# 创建文本块
# ==========================================
def create_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """创建文本块"""
    if not text:
        return []
    
    # 使用递归分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    # 过滤空块
    chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    
    return chunks

# ==========================================
# 上传接口
# ==========================================
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    kb_id: str = Form("default")
):
    tmp_path = None
    
    try:
        print(f"\n处理文件: {file.filename}")
        
        # 验证文件类型
        ext = os.path.splitext(file.filename)[-1].lower()
        allowed = [".txt", ".md", ".pdf", ".docx", ".xlsx"]
        
        if ext not in allowed:
            return {"status": "error", "msg": f"不支持的文件格式: {ext}"}
        
        # 读取并保存临时文件
        content = await file.read()
        if not content:
            return {"status": "error", "msg": "文件内容为空"}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # 提取文本
        text = extract_text_from_file(tmp_path, ext)
        
        if not text or len(text) < 10:
            return {"status": "error", "msg": f"无法提取有效文本（长度: {len(text)}）"}
        
        print(f"提取文本长度: {len(text)}")
        
        # 创建块
        chunks = create_chunks(text)
        
        if not chunks:
            return {"status": "error", "msg": "无法创建文本块"}
        
        print(f"创建了 {len(chunks)} 个文本块")
        
        # 创建 Document 对象
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk and len(chunk) > 5:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": file.filename, "chunk_id": i}
                ))
        
        if not documents:
            return {"status": "error", "msg": "没有有效的文档内容"}
        
        # 测试 embedding
        print("测试 embedding 调用...")
        test_result = embedding.embed_query("测试")
        print(f"Embedding 测试成功，向量维度: {len(test_result)}")
        
        # 创建向量库
        db_path = f"vector_store/{kb_id}"
        os.makedirs(db_path, exist_ok=True)
        
        print("创建 FAISS 向量库...")
        vectorstore = FAISS.from_documents(documents, embedding)
        print("保存向量库...")
        vectorstore.save_local(db_path)
        
        return {
            "status": "ok",
            "message": "上传成功",
            "chunks": len(chunks),
            "text_length": len(text)
        }
    
    except Exception as e:
        print(f"上传失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "msg": f"上传失败: {str(e)}"}
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ==========================================
# 检索上下文
# ==========================================
def retrieve_context(query: str, kb_id: str = "default") -> str:
    """检索相关上下文"""
    try:
        db_path = f"vector_store/{kb_id}"
        if not os.path.exists(f"{db_path}/index.faiss"):
            return ""
        
        vectorstore = FAISS.load_local(
            db_path, 
            embedding, 
            allow_dangerous_deserialization=True
        )
        
        docs = vectorstore.similarity_search(query, k=3)
        
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        return ""
    
    except Exception as e:
        print(f"检索失败: {e}")
        return ""

# ==========================================
# 聊天接口
# ==========================================
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        try:
            context = retrieve_context(request.query, request.kb_id)
            
            if context:
                system_prompt = f"""基于以下资料回答：
{context}

问题：{request.query}
请只根据资料回答。"""
            else:
                system_prompt = f"请回答：{request.query}（当前没有相关文档）"
            
            messages = [SystemMessage(content=system_prompt)]
            messages.append(HumanMessage(content=request.query))
            
            async for chunk in llm.astream(messages):
                if chunk and chunk.content:
                    yield f"data: {chunk.content}\n\n"
            
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            yield f"data: 错误: {str(e)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ==========================================
# 测试接口
# ==========================================
@app.get("/api/test/embedding")
async def test_embedding():
    """测试 embedding 是否正常工作"""
    try:
        test_text = "这是一个测试文本"
        result = embedding.embed_query(test_text)
        return {
            "status": "ok", 
            "message": "Embedding 工作正常",
            "vector_length": len(result),
            "sample": result[:5]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "message": "服务正常运行",
        "embedding_model": "qwen3-vl-embedding"
    }

@app.get("/api/kb/list")
async def list_knowledge_bases():
    """列出所有知识库"""
    kbs = []
    if os.path.exists("vector_store"):
        for item in os.listdir("vector_store"):
            item_path = os.path.join("vector_store", item)
            if os.path.isdir(item_path):
                index_file = os.path.join(item_path, "index.faiss")
                if os.path.exists(index_file):
                    kbs.append(item)
    return {"knowledge_bases": kbs}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("文档问答机器人启动")
    print("="*60)
    print(f"API Key: {DASHSCOPE_API_KEY[:10]}...{DASHSCOPE_API_KEY[-4:]}")
    print("Embedding 模型: qwen3-vl-embedding")
    print("LLM 模型: qwen-max")
    print("服务地址: http://0.0.0.0:8000")
    print("API 文档: http://0.0.0.0:8000/docs")
    print("测试接口: http://0.0.0.0:8000/api/test/embedding")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")