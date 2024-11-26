#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   zhipuai_llm.py
@Time    :   2023/10/16 22:06:26
@Author  :   0-yy-0
@Version :   1.0
@Contact :   310484121@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于智谱 AI 大模型自定义 LLM 类
'''

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
# from langchain.pydantic_v1 import Field, root_validator
from pydantic import Field, model_validator
from langchain.schema.output import GenerationChunk
from langchain.utils import get_from_dict_or_env
from self_llm import Self_LLM

logger = logging.getLogger(__name__)


class ZhipuAILLM(Self_LLM):
    """智谱AI LLM的封装类"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    client: Any  # ZhipuAI 客户端实例
    model: str = "glm-4-plus"  # 默认使用 GLM-4
    zhipuai_api_key: Optional[str] = None
    
    streaming: Optional[bool] = False
    request_timeout: Optional[int] = 60
    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95

    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """验证环境变量和初始化客户端"""
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )

        try:
            from zhipuai import ZhipuAI
            values["client"] = ZhipuAI(api_key=values["zhipuai_api_key"])
        except ImportError:
            raise ValueError(
                "zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取模型标识参数"""
        return {
            **{"model": self.model},
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "zhipuai"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "stream": self.streaming,
            "top_p": self.top_p,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行同步调用"""
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self._default_params,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error during API call: {str(e)}")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行异步调用"""
        if self.streaming:
            completion = ""
            async for chunk in self._astream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.client.chat.asyncCompletions.create(
                model=self.model,
                messages=messages,
                **self._default_params,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error during async API call: {str(e)}")

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """处理流式输出"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **self._default_params,
                **kwargs
            )

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    chunk_text = chunk.choices[0].delta.content
                    yield GenerationChunk(text=chunk_text)
                    if run_manager:
                        run_manager.on_llm_new_token(chunk_text)
        except Exception as e:
            raise ValueError(f"Error during streaming: {str(e)}")

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """处理异步流式输出"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.client.chat.asyncCompletions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **self._default_params,
                **kwargs
            )

            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    chunk_text = chunk.choices[0].delta.content
                    yield GenerationChunk(text=chunk_text)
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk_text)
        except Exception as e:
            raise ValueError(f"Error during async streaming: {str(e)}")
