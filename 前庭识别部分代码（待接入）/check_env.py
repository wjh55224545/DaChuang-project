# check_env.py
import sys
print(f"Python版本: {sys.version}")

packages = ['torch', 'numpy', 'pandas', 'sklearn', 'openpyxl', 'matplotlib', 'tqdm']
for pkg in packages:
    try:
        if pkg == 'sklearn':
            __import__('sklearn')
        else:
            __import__(pkg)
        print(f"✓ {pkg} 已安装")
    except ImportError:
        print(f"✗ {pkg} 未安装")