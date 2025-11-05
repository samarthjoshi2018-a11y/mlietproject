from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    
    requirement_lst: List[str] = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found.")   
    return requirement_lst



setup(
    name='my_package',
    version='0.1.0',
    author='Samarth joshi',
    author_email="samarthjoshi2018@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

