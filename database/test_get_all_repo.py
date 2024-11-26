import json
import requests
import os
import base64
import loguru
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
# 从环境变量中获取TOKEN
TOKEN = os.getenv('TOKEN')
# 在脚本开始处添加
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
# 定义获取组织仓库的函数
def get_repos(org_name, token, export_dir):
    """获取组织仓库的函数"""
    # 清除系统代理设置
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    try:
        url = f'https://api.github.com/orgs/{org_name}/repos'
        # 禁用代理，添加超时设置
        response = requests.get(
            url, 
            headers=headers, 
            params={'per_page': 200, 'page': 0},
            proxies={},  # 禁用代理
            timeout=30,  # 添加超时
            verify=True  # SSL验证
        )
        
        if response.status_code == 200:
            repos = response.json()
            loguru.logger.info(f'Fetched {len(repos)} repositories for {org_name}.')
            
            # 确保导出目录存在
            os.makedirs(export_dir, exist_ok=True)
            
            # 保存仓库列表
            repositories_path = os.path.join(export_dir, 'repositories.txt')
            with open(repositories_path, 'w', encoding='utf-8') as file:
                for repo in repos:
                    file.write(repo['name'] + '\n')
            return repos
            
        else:
            loguru.logger.error(f"Error fetching repositories: {response.status_code}")
            loguru.logger.error(response.text)
            return []
            
    except requests.exceptions.ProxyError:
        loguru.logger.error("Proxy error encountered. Trying without proxy...")
        # 如果出现代理错误，设置NO_PROXY
        os.environ['NO_PROXY'] = '*'
        # 重试请求
        return get_repos(org_name, token, export_dir)
        
    except requests.exceptions.RequestException as e:
        loguru.logger.error(f"Request error: {str(e)}")
        return []
        
    except Exception as e:
        loguru.logger.error(f"Unexpected error: {str(e)}")
        return []
# 定义拉取仓库README文件的函数
def fetch_repo_readme(org_name, repo_name, token, export_dir):
    """拉取仓库README文件的函数"""
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    try:
        url = f'https://api.github.com/repos/{org_name}/{repo_name}/readme'
        # 同样禁用代理
        response = requests.get(
            url, 
            headers=headers,
            proxies={},
            timeout=30,
            verify=True
        )
        
        if response.status_code == 200:
            readme_content = response.json()['content']
            readme_content = base64.b64decode(readme_content).decode('utf-8')
            
            repo_dir = os.path.join(export_dir, repo_name)
            os.makedirs(repo_dir, exist_ok=True)
            
            readme_path = os.path.join(repo_dir, 'README.md')
            with open(readme_path, 'w', encoding='utf-8') as file:
                file.write(readme_content)
                
        else:
            loguru.logger.error(f"Error fetching README for {repo_name}: {response.status_code}")
            loguru.logger.error(response.text)
            
    except Exception as e:
        loguru.logger.error(f"Error fetching README for {repo_name}: {str(e)}")
# 主函数
if __name__ == '__main__':
    # 配置组织名称
    org_name = 'datawhalechina'
    # 配置 export_dir
    export_dir = "E:/CATL/2.项目与比赛/llm_race/templates"  # 请替换为实际的目录路径
    # 获取仓库列表
    repos = get_repos(org_name, TOKEN, export_dir)
    # 打印仓库名称
    if repos:
        for repo in repos:
            repo_name = repo['name']
            # 拉取每个仓库的README
            # fetch_repo_readme(org_name, repo_name, TOKEN, export_dir)
            print(repo_name)
    # 清理临时文件夹
    # if os.path.exists('temp'):
    #     shutil.rmtree('temp')