import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tempfile
from dotenv import load_dotenv, find_dotenv
from embedding.call_embedding import get_embedding
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.document_loaders import PDFPlumberLoader, UnstructuredPDFLoader
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
# from langchain_chroma  import Chroma
# 首先实现基本配置

DEFAULT_DB_PATH = "./knowledge_db"
DEFAULT_PERSIST_PATH = "./vector_db"


def get_files(dir_path):
    """递归获取目录下所有支持的文件"""
    file_list = []
    supported_extensions = {'.txt', '.md', '.pdf', '.doc', '.docx'}
    
    for root, _, files in os.walk(dir_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                file_list.append(os.path.join(root, file))
    return file_list

def create_db_info(files=DEFAULT_DB_PATH, embeddings="m3e", persist_directory=DEFAULT_PERSIST_PATH):
    """
    创建向量数据库的包装函数，返回处理状态
    """
    try:
        if not files:
            return "⚠️ 请先上传文件！"
            
        if embeddings not in ['openai', 'm3e', 'zhipuai']:
            return f"⚠️ 不支持的 Embedding 模型: {embeddings}，请选择 openai、m3e 或 zhipuai"
            
        print(f"开始处理文件，使用 {embeddings} 模型...")
        vectordb = create_db(files, persist_directory, embeddings)
        
        # 确保返回字符串消息
        success_msg = "✅ 知识库创建成功！文件已成功向量化并存储。"
        print(success_msg)  # 在控制台打印日志
        return success_msg
        
    except Exception as e:
        error_msg = f"❌ 知识库创建失败！错误信息：{str(e)}"
        print(error_msg)  # 在控制台打印错误日志
        return error_msg


def create_db(files, persist_directory, embeddings_model_name):
    """
    创建向量数据库
    Args:
        files: 文件路径或目录路径列表，或Gradio上传的文件对象列表
        persist_directory: 向量库持久化路径
        embeddings_model_name: embedding 模型名称 ("openai", "m3e" 或 "zhipuai")
    Returns:
        vectordb: 向量数据库
    """
    docs = []
    
    # 如果传入的是单个文件/路径，转换为列表
    if not isinstance(files, list):
        files = [files]
    
    # 获取所有文件路径
    all_files = []
    for file_path in files:
        # 处理Gradio上传的文件对象
        if hasattr(file_path, 'name'):
            all_files.append(file_path.name)
        # 处理普通路径字符串
        elif isinstance(file_path, (str, bytes, os.PathLike)):
            if os.path.isdir(file_path):
                print(f"正在扫描目录: {file_path}")
                all_files.extend(get_files(file_path))
            else:
                all_files.append(file_path)
    
    print(f"找到 {len(all_files)} 个文件需要处理")
    
    # 处理每个文件
    for file in all_files:
        try:
            print(f"Processing file: {file}")
            
            if file.lower().endswith('.txt'):
                loader = TextLoader(file, encoding='utf-8')
                current_docs = loader.load()
                print(f"Loaded {len(current_docs)} documents from txt file")
                docs.extend(current_docs)
                
            elif file.lower().endswith('.md'):
                loader = UnstructuredMarkdownLoader(file)
                current_docs = loader.load()
                print(f"Loaded {len(current_docs)} documents from md file")
                docs.extend(current_docs)
                
            elif file.lower().endswith('.pdf'):
                try:
                    loader = PyPDFLoader(file)
                    current_docs = loader.load()
                    print(f"Loaded {len(current_docs)} documents from pdf file")
                    docs.extend(current_docs)
                except Exception as pdf_error:
                    print(f"Error loading PDF with PyPDFLoader: {str(pdf_error)}")
                    try:
                        import pdfplumber
                        with pdfplumber.open(file) as pdf:
                            for i, page in enumerate(pdf.pages):
                                text = page.extract_text()
                                if text:
                                    docs.append(Document(page_content=text, metadata={"source": file}))
                            print(f"Loaded {len(pdf.pages)} pages with PDFPlumber")
                    except Exception as plumber_error:
                        print(f"Error loading PDF with PDFPlumber: {str(plumber_error)}")
                        continue
                        
            elif file.lower().endswith('.docx'):
                try:
                    # 首先尝试使用 Docx2txtLoader
                    loader = Docx2txtLoader(file)
                    current_docs = loader.load()
                    print(f"Loaded {len(current_docs)} documents from docx file")
                    docs.extend(current_docs)
                except Exception as docx_error:
                    print(f"Error loading DOCX with Docx2txtLoader: {str(docx_error)}")
                    try:
                        # 如果失败，尝试使用 UnstructuredWordDocumentLoader
                        loader = UnstructuredWordDocumentLoader(file)
                        current_docs = loader.load()
                        print(f"Loaded {len(current_docs)} documents from docx file using UnstructuredLoader")
                        docs.extend(current_docs)
                    except Exception as unstructured_error:
                        print(f"Error loading DOCX with UnstructuredLoader: {str(unstructured_error)}")
                        continue
                        
            elif file.lower().endswith('.doc'):
                try:
                    # 对于旧版 .doc 文件使用 UnstructuredWordDocumentLoader
                    loader = UnstructuredWordDocumentLoader(file)
                    current_docs = loader.load()
                    print(f"Loaded {len(current_docs)} documents from doc file")
                    docs.extend(current_docs)
                except Exception as doc_error:
                    print(f"Error loading DOC file: {str(doc_error)}")
                    continue
                    
            else:
                print(f"Skipping unsupported file type: {file}")
                continue
                
        except Exception as e:
            print(f"Error loading file {file}: {str(e)}")
            continue
    
    print(f"Total documents loaded: {len(docs)}")
    
    if not docs:
        raise ValueError("No documents were loaded. Please check your input files and their formats.")
    
    # 切分文档
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")
    
    # 初始化 embedding 模型
    embedding_model = get_embedding(embeddings_model_name)
    
    # 创建向量数据库时使用初始化好的 embedding 模型
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,  # 使用初始化后的 embedding 模型
        persist_directory=persist_directory
    )
    vectordb.persist()
    
    return vectordb


def presit_knowledge_db(vectordb):
    """
    该函数用于持久化向量数据库。

    参数:
    vectordb: 要持久化的向量数据库。
    """
    vectordb.persist()


def load_knowledge_db(path, embeddings):
    """
    该函数用于加载向量数据库。

    参数:
    path: 要加载的向量数据库路径。
    embeddings: 向量数据库使用的 embedding 模型。

    返回:
    vectordb: 加载的数据库。
    """
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb


if __name__ == "__main__":
    create_db(embeddings="m3e")
