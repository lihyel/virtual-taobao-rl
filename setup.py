from setuptools import setup, find_packages  # 중복 제거 및 find_packages 포함

setup(
    name='virtualTB',
    version='0.0.5',
    # gym 버전을 0.21.0으로 고정해야 에러가 안 남
    install_requires=[
        'gym==0.21.0', 
        'torch>=0.4.0',
        'numpy>=1.18.0',
        'pandas',
        'scikit-learn'
    ],
    # virtualTB 폴더와 그 하위 폴더들만 패키지로 포함
    packages=find_packages(include=['virtualTB', 'virtualTB.*']),
)