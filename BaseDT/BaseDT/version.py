import os

__version__='0.1.3'
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
    ____                  ____  ______
   / __ )____ _________  / __ \/_  __/
  / __  / __ `/ ___/ _ \/ / / / / /   
 / /_/ / /_/ (__  )  __/ /_/ / / /    
/_____/\__,_/____/\___/_____/ /_/                                                                          
    """)
    print("BaseDT 是一个功能强大且易于扩展的数据处理工具。")
    print("BaseDT is a powerful and easily extensible data processing tool.")
    print("相关网址：")
    print("-文档网址 :  https://xedu.readthedocs.io")
    print("-官网网址 :  https://www.openinnolab.org.cn/pjEdu/xedu/baseedu")


version_info = parse_version_info(__version__)
# path_info = parse_version_info(__path__)
