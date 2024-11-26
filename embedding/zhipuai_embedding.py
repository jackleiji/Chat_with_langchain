from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """智谱AI的文本嵌入模型封装"""

    zhipuai_api_key: Optional[str] = None
    client: Any = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """验证环境变量并初始化客户端"""
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )

        try:
            import zhipuai
            # 直接使用 zhipuai
            zhipuai.api_key = values["zhipuai_api_key"]
            values["client"] = zhipuai
        except ImportError:
            raise ValueError(
                "Zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return values

    def _embed(self, texts: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        try:
            resp = self.client.invoke(
                model="text_embedding",
                prompt=texts
            )
            
            if resp["code"] != 200:
                raise ValueError(
                    f"API call failed with code {resp['code']}: {resp.get('msg')}"
                )
                
            return resp["data"]["embedding"]
            
        except Exception as e:
            raise ValueError(f"Error during embedding generation: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """获取多个文档的嵌入向量"""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档（暂不支持）"""
        raise NotImplementedError(
            "Async embedding is not supported by Zhipuai API"
        )

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询（暂不支持）"""
        raise NotImplementedError(
            "Async embedding is not supported by Zhipuai API"
        )
