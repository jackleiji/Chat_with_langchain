from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
# from langchain_chroma  import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import sys
sys.path.append('/Users/lta/Desktop/llm-universe/project')
sys.path.append("E:/CATL/2.项目与比赛/Chat_with_Datawhale_langchain")
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re
import os
import shutil

class Chat_QA_chain_self:
    """"
    带历史记录的问答链  
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）  
    """
    def __init__(self, model:str=None, temperature:float=0.0, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None, Wenxin_secret_key:str=None, embedding:str="m3e", embedding_key:str=None):
        """
        初始化方法
        """
        try:
            # 清除旧的向量库
            # if os.path.exists(persist_path):
            #     shutil.rmtree(persist_path)
            #     os.makedirs(persist_path)
            
            self.model = model
            self.temperature = temperature
            self.top_k = top_k
            self.chat_history = chat_history
            self.file_path = file_path
            self.persist_path = persist_path
            self.appid = appid
            self.api_key = api_key
            self.Spark_api_secret = Spark_api_secret
            self.Wenxin_secret_key = Wenxin_secret_key
            self.embedding = embedding
            self.embedding_key = embedding_key
            
            # 获取向量数据库
            self.vectordb = get_vectordb(
                file_path=self.file_path,
                persist_path=self.persist_path,
                embedding=self.embedding,
                embedding_key=self.embedding_key
            )
            
        except Exception as e:
            raise Exception(f"Error initializing Chat_QA_chain_self: {str(e)}")

    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()

    
    def change_history_length(self,history_len:int=1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

 
    def answer(self, question: str = None, temperature = None, top_k = 4):
        """
        核心方法，调用问答链
        
        Args:
            question: 用户提问
            temperature: 温度系数
            top_k: 返回检索的前k个相似文档
            
        Returns:
            tuple: (answer, chat_history) 包含回答和更新后的对话历史
        """
        try:
            if not question or len(question.strip()) == 0:
                return "", self.chat_history
            
            if temperature is None:
                temperature = self.temperature
            
            llm = model_to_llm(
                self.model, 
                temperature, 
                self.appid, 
                self.api_key, 
                self.Spark_api_secret,
                self.Wenxin_secret_key
            )

            retriever = self.vectordb.as_retriever(
                search_type="similarity",   
                search_kwargs={'k': top_k}
            )

            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever
            )
            
            result = qa({"question": question, "chat_history": self.chat_history})
            answer = result['answer']
            answer = re.sub(r"\\n", '<br/>', answer)
            self.chat_history.append((question, answer))
            
            return answer, self.chat_history  # 返回元组 (answer, chat_history)
            
        except Exception as e:
            error_msg = f"Error during QA process: {str(e)}"
            return error_msg, self.chat_history
















