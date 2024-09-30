import os

__version__='0.0.4'
__path__=os.path.abspath(os.getcwd())

def parse_version_info(version_str):
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)

def hello():
                                                 
    print("""
    ____                  ____             __           
   / __ )____ _________  / __ \___  ____  / /___  __  __
  / __  / __ `/ ___/ _ \/ / / / _ \/ __ \/ / __ \/ / / /
 / /_/ / /_/ (__  )  __/ /_/ /  __/ /_/ / / /_/ / /_/ / 
/_____/\__,_/____/\___/_____/\___/ .___/_/\____/\__, /  
                                /_/            /____/                                         
    """)
    print("BaseDeploy 是一款简单易用的模型部署工具。")
    print("BaseDeploy is an simple deployment tool.")
    print("相关网址：")
    print("-文档网址 :  https://xedu.readthedocs.io")
    print("-官网网址 :  https://www.openinnolab.org.cn/pjEdu/xedu/baseedu")


version_info = parse_version_info(__version__)
# path_info = parse_version_info(__path__)
