# 导入必要的库

import sys
import os                # 用于操作系统相关的操作，例如读取环境变量

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import IPython.display   # 用于在 IPython 环境中显示数据，例如图片
import io                # 用于处理流式数据（例如文件流）
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from database.create_db import create_db_info
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
import re
import warnings
 
warnings.filterwarnings("ignore")
# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量
_ = load_dotenv(find_dotenv())
LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    # "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
    "zhipuai": ["glm-4-plus", "glm-4-air", "chatglm_lite"]
}


LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
INIT_LLM = "glm-4-plus"
EMBEDDING_MODEL_LIST = ['zhipuai', 'm3e']
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = "./knowledge_db"
DEFAULT_PERSIST_PATH = "./vector_db"
AIGC_AVATAR_PATH = "./figures/aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "./figures/datawhale_avatar.png"
AIGC_LOGO_PATH = "./figures/aigc_logo.png"
DATAWHALE_LOGO_PATH = "./figures/datawhale_logo.png"
HEAD_IMAGE_PATH = "./figures/head.png"

def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")

class Model_center():
    """
    存储问答 Chain 的对象 

    - chat_qa_chain_self: 以 (model, embedding) 为键存储的带历史记录的问答链。
    - qa_chain_self: 以 (model, embedding) 为键存储的不带历史记录的问答链。
    """
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(self, question, chat_history: list = [], model="glm-4-plus", embedding="zhipuai", temperature=0.0, top_k=4, history_len=3, file_path=DEFAULT_DB_PATH, persist_path=DEFAULT_PERSIST_PATH):
        """带历史记录的问答"""
        if not question or len(str(question).strip()) == 0:
            return "", chat_history
        
        try:
            progress = gr.Progress()
            progress(0, desc="正在准备模型...")
            
            if (model, embedding) not in self.chat_qa_chain_self:
                progress(0.2, desc="正在加载模型...")
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    chat_history=chat_history,
                    file_path=file_path,
                    persist_path=persist_path,
                    embedding=embedding
                )
            
            chain = self.chat_qa_chain_self[(model, embedding)]
            progress(0.5, desc="正在生成回答...")
            answer = chain.answer(question=str(question), temperature=temperature, top_k=top_k)
            
            if not isinstance(answer, str):
                answer = str(answer)
            
            chat_history.append((str(question), answer))
            progress(1.0, desc="完成!")
            return "", chat_history
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            if chat_history is None:
                chat_history = []
            chat_history.append((str(question), error_message))
            return "", chat_history

    def qa_chain_self_answer(self, question, chat_history: list = [], model="openai", embedding="openai", temperature=0.0, top_k=4, file_path=DEFAULT_DB_PATH, persist_path=DEFAULT_PERSIST_PATH):
        """不带历史记录的问答"""
        if not question or len(str(question).strip()) == 0:
            return "", chat_history
            
        try: 
            progress = gr.Progress()
            progress(0, desc="正在准备模型...")
            
            # 获取 API key
            # model_key = parse_llm_api_key(model)  # 使用导入的函数
            
            if (model, embedding) not in self.qa_chain_self:
                progress(0.2, desc="正在加载模型...")
                self.qa_chain_self[(model, embedding)] = QA_chain_self(
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    file_path=file_path,
                    persist_path=persist_path,
                    embedding=embedding
                )
            
            chain = self.qa_chain_self[(model, embedding)]
            progress(0.5, desc="正在生成回答...")
            answer = chain.answer(question=str(question), temperature=temperature, top_k=top_k)
            
            if not isinstance(answer, str):
                answer = str(answer)
            
            chat_history.append((str(question), answer))
            progress(1.0, desc="完成!")
            return "", chat_history
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            if chat_history is None:
                chat_history = []
            chat_history.append((str(question), error_message))
            return "", chat_history

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()


def format_chat_prompt(message, chat_history):
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = "你是PHM平台的小艾助手"
    # 遍历聊天历史记录。
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式后的 prompt。
    return prompt



def respond(message, chat_history, llm, history_len=3, temperature=0.1):
    """处理用户消息并返回响应"""
    if not message or len(message.strip()) == 0:
        return "", chat_history
    try:
        # 确保 chat_history 是列表
        if chat_history is None:
            chat_history = []
            
        # 格式化聊天记录
        prompt = format_chat_prompt(message, chat_history[-history_len:] if history_len > 0 else [])
        
        # 获取 AI 回复
        bot_message = get_completion(
            prompt=prompt,
            model=llm,
            temperature=temperature
        )
        
        # 确保返回的是字符串类型
        if not isinstance(bot_message, str):
            bot_message = str(bot_message)
            
        # 添加到聊天历史
        chat_history.append((str(message), bot_message))
        
        return "", chat_history
    except Exception as e:
        error_message = f"Error: {str(e)}"
        chat_history.append((str(message), error_message))
        return "", chat_history


model_center = Model_center()

# 修改 CSS 样式
CSS = """
/* 灰色背景 */
.gradio-container {
    background-color: #f5f5f5 !important;
    max-width: 1500px !important;  /* 设置最大宽度 */
    margin: 0 auto !important;     /* 水平居中 */
    padding: 20px !important;      /* 添加内边距 */
}

/* 内容区域样式 */
.contain {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    padding: 16px;
    margin: 16px 0;
    width: 100%;  /* 确保内容区域填充容器宽度 */
}

/* 聊天窗口样式 */
.custom-chatbot {
    font-size: 10.5pt !important;
    line-height: 1.5 !important;
    width: 100% !important;        /* 确保聊天窗口填充容器宽度 */
    max-width: 1200px !important;   /* 聊天窗口最大宽度 */
    margin: 0 auto !important;     /* 聊天窗口居中 */
}

/* 调整列布局 */
.main-cols {
    gap: 20px !important;          /* 列之间的间距 */
    align-items: start !important; /* 列顶部对齐 */
}

/* 右侧控制面板样式 */
.control-panel {
    background-color: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    width: 100%;                   /* 填充列宽度 */
}

/* 页脚样式 */
.footer-container {
    display: flex;
    justify-content: space-between;
    padding: 20px;
    background-color: white;
    border-radius: 8px;
    margin-top: 20px;
    width: 100%;                   /* 确保页脚填充容器宽度 */
    max-width: 1160px !important;  /* 页脚最大宽度 */
    margin-left: auto !important;
    margin-right: auto !important;
}

/* 炫酷的标题样式 */
.artistic-title {
    font-family: 'Arial', sans-serif;
    font-size: 2.5em;
    font-weight: bold;
    background: linear-gradient(120deg, 
        #1a237e 0%, 
        #0d47a1 25%,
        #0288d1 50%,
        #0097a7 75%,
        #00838f 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
    position: relative;
    padding: 10px 0;
}

.artistic-title::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 25%;
    width: 50%;
    height: 2px;
    background: linear-gradient(90deg, 
        transparent 0%,
        #2196F3 50%,
        transparent 100%
    );
}

.subtitle {
    font-family: 'Arial', sans-serif;
    color: #455a64;
    font-size: 1.2em;
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-top: 10px;
    background: linear-gradient(45deg, #455a64, #78909c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
"""

# 创建 Gradio 界面
with gr.Blocks(css=CSS) as demo:
    # Add header image
    gr.Image(value=HEAD_IMAGE_PATH, scale=1, show_label=False, show_download_button=False)
    
    # Existing logo row
    with gr.Row(equal_height=True, elem_classes="main-cols"):           
        gr.Image(value=AIGC_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)
   
        with gr.Column(scale=2):
            gr.Markdown("""
                <div style="text-align: center; padding: 20px 0;">
                    <h1 class="artistic-title">基于LLM的智能专利检索助理</h1>
                    <div class="subtitle">LLM-UNIVERSE</div>
                </div>
                """)
        gr.Image(value=DATAWHALE_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)

    with gr.Row():
        with gr.Column(scale=4):
            # 修改聊天窗口配置
            chatbot = gr.Chatbot(
                height=600,  # 调整高度为500px
                show_copy_button=True, 
                show_share_button=True, 
                avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH),
                elem_classes="custom-chatbot"  # 应用自定义样式
            )
            
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                button_state = gr.State("")  # 添加状态件来跟踪当前选中的按钮
                db_with_his_btn = gr.Button(
                    "Chat db with history",
                    elem_classes=["custom-button"],
                    variant="primary"
                )
                db_wo_his_btn = gr.Button(
                    "Chat db without history",
                    elem_classes=["custom-button"],
                    variant="primary"
                )
                llm_btn = gr.Button(
                    "Chat with llm",
                    elem_classes=["custom-button"],
                    variant="primary"
                )
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        with gr.Column(scale=1, elem_classes="control-panel"):
            # 修改文件选择器配置
            file = gr.File(
                label='请选择知识库文件',
                file_count='multiple',
                file_types=['.txt', '.md', '.docx', '.doc', '.pdf'],
                show_label=True,
                container=True,
                scale=1,
                min_width=None,
                interactive=True,
                visible=True,
                elem_id="file_upload"
            )
            
            with gr.Row():
                init_db = gr.Button("知识库文件向量化", variant="primary")
                init_status = gr.Textbox(
                    label="向量化状态",
                    placeholder="请点击上方按钮开始向量化...",
                    interactive=False,
                    show_label=True
                )
            
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)

                top_k = gr.Slider(1,
                                  10,
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_LLM,
                    interactive=True)

                embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                         label="Embedding model",
                                         value=INIT_EMBEDDING_MODEL)

        # 修改按钮点击事件的处理方式
        # 按钮状态切换事件
        db_with_his_btn.click(
            fn=model_center.chat_qa_chain_self_answer,
            inputs=[msg, chatbot, llm, embeddings, temperature, top_k, history_len],
            outputs=[msg, chatbot],
            show_progress="full"
        )

        db_wo_his_btn.click(
            fn=model_center.qa_chain_self_answer,
            inputs=[msg, chatbot, llm, embeddings, temperature, top_k],
            outputs=[msg, chatbot],
            show_progress="full"
        )

        llm_btn.click(
            fn=respond,
            inputs=[msg, chatbot, llm, history_len, temperature],
            outputs=[msg, chatbot],
            show_progress="full"
        )

        # 初始化数据库按钮点击事件
        init_db.click(
            fn=create_db_info,
            inputs=[file, embeddings],
            outputs=init_status
        )

        msg.submit(
            respond,
            inputs=[msg, chatbot, llm, history_len, temperature],
            outputs=[msg, chatbot],
            show_progress="full"
        )

        clear.click(model_center.clear_history)

    # 添加页脚息
    gr.Markdown(
        """
        <div class="footer-container">
            <div class="footer-left">
                <div class="footer-content">提醒：</div>
                <div class="footer-content">1. 使用时请先上传自己的知识文件，不然将会解析项目自带的知识库。</div>
                <div class="footer-content">2. 初始化数据库时间可能较长，请耐等待。</div>
                <div class="footer-content">3. 使用中如果出现异常，将会在文本入框进行展示，请不要惊慌。</div>
            </div>
            <div class="footer-right">
                <div class="footer-content">🤖 LLM智能助手 - 您的智能文献检索专家</div>
                <div class="footer-content">📚 快速、准确、高效 - 提升您的文献检索效率</div>
                <div class="footer-content">💡 支持多种文献格式，智能分析，精准答复</div>
            </div>
        </div>
        """)

# 添加JavaScript代码来处理按钮状态
js_code = """
function toggleButton(btnId) {
    const buttons = document.querySelectorAll('.custom-button');
    let selectedText = '';
    
    buttons.forEach(btn => {
        if (btn.id === btnId) {
            const wasSelected = btn.classList.contains('selected');
            btn.classList.toggle('selected');
            if (!wasSelected) {
                selectedText = '已选择: ' + btn.textContent;
            }
        } else {
            btn.classList.remove('selected');
        }
    });
    
    // 显示提示信息
    const toast = document.createElement('div');
    toast.className = 'toast-message';
    toast.textContent = selectedText;
    document.body.appendChild(toast);
    
    // 2秒后移除提示
    setTimeout(() => {
        toast.classList.add('toast-fade-out');
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 2000);
    
    return true;
}
"""

# 启动应用
demo.launch(
    share=False,
    server_port=7860,
    server_name="127.0.0.1",
    favicon_path="./figures/aigc_logo.png"  # 保留支持的参数
)
