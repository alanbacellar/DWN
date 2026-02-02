import os
import torch
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cpp_dir = os.path.join('src', 'torch_dwn', 'custom_operators', 'cpp')
cuda_dir = os.path.join('src', 'torch_dwn', 'custom_operators', 'cuda')

ext_modules = []

# C++
if os.path.exists(cpp_dir):
    for filename in os.listdir(cpp_dir):
        if filename.endswith('.cpp'):
            module_name = filename[:-4]
            source_path = os.path.join(cpp_dir, filename)
            ext_modules.append(
                CppExtension(
                    name=module_name, 
                    sources=[source_path]
                )
            )

# CUDA
if os.path.exists(cuda_dir):
    for filename in os.listdir(cuda_dir):
        if filename.endswith('.cpp'):
            module_name = filename[:-4]
            kernel_filename = module_name + '_kernel.cu'
            cpp_source = os.path.join(cuda_dir, filename)
            cuda_source = os.path.join(cuda_dir, kernel_filename)
            if os.path.exists(cuda_source):
                ext_modules.append(
                    CUDAExtension(
                        name=module_name, 
                        sources=[cpp_source, cuda_source]
                    )
                )

setup(
    name='torch_dwn',
    version="1.1.1",
    author="Alan T. L. Bacellar",
    author_email="alanbacellar@gmail.com",
    description="Differentiable Weightless Neural Networks (DWN) PyTorch Module",
    url="https://github.com/alanbacellar/DWN",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    
    install_requires=[
        'torch'
    ],
    include_package_data=True,
)