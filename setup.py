import os
import torch
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cpp_dir = os.path.join('src', 'torch_dwn', 'custom_operators', 'cpp')
cuda_dir = os.path.join('src', 'torch_dwn', 'custom_operators', 'cuda')

ext_modules = []

for filename in os.listdir(cpp_dir):
    if filename[-3:] == 'cpp':
        ext_modules.append(CppExtension(filename[:-3], [os.path.join(cpp_dir, filename)]))

for filename in os.listdir(cuda_dir):
    if filename[-3:] == 'cpp':
        module_name = filename[:-4]
        kernel_filename = module_name + '_kernel.cu'
        ext_modules.append(CUDAExtension(module_name, [os.path.join(cuda_dir, filename), os.path.join(cuda_dir, kernel_filename)]))

setup(
    name='torch_dwn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    version="1.0.9",
    author="Alan T. L. Bacellar",
    author_email="alanbacellar@gmail.com",
    description="Differentiable Weightless Neural Networks (DWN) PyTorch Module",
    url="https://github.com/alanbacellar/DWN",
    install_requires=[
        'torch'
    ]
)
