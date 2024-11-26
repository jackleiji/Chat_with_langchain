import sys
from typing import Optional, Any, List
from zhipuai import ZhipuAI
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, BaseModel
from llm.call_llm import parse_llm_api_key

class ZhipuAIWrapper(LLM, BaseModel):
    """自定义智谱AI包装器"""
    
    model: str = Field(..., description="Model name")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    api_key: Optional[str] = Field(default=None, description="ZhipuAI API key")
    client: Any = Field(default=None, description="ZhipuAI client")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = ZhipuAI(api_key=self.api_key) if self.api_key else None
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用智谱AI API"""
        if not self.client:
            raise ValueError("ZhipuAI client not initialized. Please provide api_key.")
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling ZhipuAI: {str(e)}")
            
    @property
    def _llm_type(self) -> str:
        return "zhipuai"

def model_to_llm(model:str=None, temperature:float=0.0, appid:str=None, api_key:str=None, Spark_api_secret:str=None, Wenxin_secret_key:str=None):
    """
    根据模型名称返回对应的LLM实例
    """
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        if api_key == None:
            api_key = parse_llm_api_key("openai")
        return ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key)
        
    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        if api_key == None or Wenxin_secret_key == None:
            api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
        return Wenxin_LLM(model=model, temperature=temperature, api_key=api_key, secret_key=Wenxin_secret_key)
        
    elif model in ["Spark-1.5", "Spark-2.0"]:
        if api_key == None or appid == None and Spark_api_secret == None:
            api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
        return Spark_LLM(model=model, temperature=temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)
        
    elif model in ["glm-4", "glm-4-plus", "chatglm_std", "chatglm_lite"]:
        if api_key == None:
            api_key = parse_llm_api_key("zhipuai")
        return ZhipuAIWrapper(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        
    else:
        raise ValueError(f"Model {model} not supported!")