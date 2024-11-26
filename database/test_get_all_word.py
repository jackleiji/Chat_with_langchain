import os
import shutil
import loguru
from docx import Document
from pathlib import Path

def get_word_files(source_dir, export_dir):
    """
    获取指定目录下的所有Word文档
    
    Args:
        source_dir: 源文件夹路径
        export_dir: 导出文件夹路径
    
    Returns:
        list: 包含所有Word文档信息的列表
    """
    # 确保导出目录存在
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        
    # 记录找到的Word文件
    word_files = []
    
    # 使用 repositories.txt 记录文件列表
    repositories_path = os.path.join(export_dir, 'word_files.txt')
    
    try:
        # 遍历源目录
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.doc', '.docx')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, source_dir)
                    word_files.append({
                        'name': file,
                        'path': file_path,
                        'relative_path': relative_path
                    })
        
        loguru.logger.info(f'Found {len(word_files)} Word documents.')
        
        # 保存文件列表
        with open(repositories_path, 'w', encoding='utf-8') as file:
            for word_file in word_files:
                file.write(f"{word_file['name']}\n")
        
        return word_files
        
    except Exception as e:
        loguru.logger.error(f"Error scanning Word files: {str(e)}")
        return []

def fetch_word_content(word_file, export_dir):
    """
    提取Word文档内容并保存
    
    Args:
        word_file: Word文件信息字典
        export_dir: 导出目录
    """
    try:
        # 创建目标目录
        doc_dir = os.path.join(export_dir, Path(word_file['relative_path']).stem)
        if not os.path.exists(doc_dir):
            os.makedirs(doc_dir)
            
        # 目标文件路径
        export_path = os.path.join(doc_dir, word_file['name'])
        
        # 复制文件
        shutil.copy2(word_file['path'], export_path)
        
        # 提取文本内容（可选）
        try:
            doc = Document(word_file['path'])
            text_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            # 保存文本内容
            text_path = os.path.join(doc_dir, f"{Path(word_file['name']).stem}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
                
        except Exception as e:
            loguru.logger.error(f"Error extracting text from {word_file['name']}: {str(e)}")
            
    except Exception as e:
        loguru.logger.error(f"Error processing {word_file['name']}: {str(e)}")

def main():
    # 配置源目录和导出目录
    source_dir = "E:/CATL/入职指南"  # 请替换为实际的源目录路径
    export_dir = "E:/CATL/入职指南/word_db"  # 请替换为实际的导出目录路径
    
    # 获取Word文件列表
    word_files = get_word_files(source_dir, export_dir)
    
    # 处理每个Word文件
    if word_files:
        for word_file in word_files:
            loguru.logger.info(f"Processing: {word_file['name']}")
            fetch_word_content(word_file, export_dir)
            
    # 清理临时文件（如果需要）
    # if os.path.exists('temp'):
    #     shutil.rmtree('temp')

if __name__ == '__main__':
    main() 