import sys 
# sys.path.append("../embedding") 
# sys.path.append("../database") 

from langchain.embeddings.openai import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型
import os
from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from database.create_db import create_db,load_knowledge_db
from embedding.call_embedding import get_embedding

def get_vectordb(file_path: str, persist_path: str, embedding: str = "m3e", embedding_key: str = ""):
    """
    获取向量数据库，如果不存在则创建
    Args:
        file_path: 文档路径
        persist_path: 向量库持久化路径
        embedding: embedding 模型名称
        embedding_key: embedding 模型密钥
    Returns:
        vectordb: 向量数据库
    """
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            #print("目录为空")
            vectordb = create_db(file_path, persist_path, embedding)
            #presit_knowledge_db(vectordb)
            vectordb = load_knowledge_db(persist_path, embedding)
        else:
            #print("目录不为空")
            vectordb = load_knowledge_db(persist_path, embedding)
    else: #目录不存在，从头开始创建向量数据库
        vectordb = create_db(file_path, persist_path, embedding)
        #presit_knowledge_db(vectordb)
        vectordb = load_knowledge_db(persist_path, embedding)

    return vectordb