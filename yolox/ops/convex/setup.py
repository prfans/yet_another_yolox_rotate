# coding: utf-8

import os
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"

__version__ = '0.0.1'


def make_cuda_ext(name, sources, with_cuda=True):
    define_macros = []
    extra_compile_args = {'cxx': ["-O2", "-std=c++14", "-Wall"]}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1' and with_cuda:
        define_macros += [('WITH_CUDA', None)] # 宏定义的声明，#define WITH_CUDA None
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-O2',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
    else:
        print(f"Compiling {name} without CUAD")
        extension = CppExtension

    return extension(
        name=name,
        sources=sources,
        define_macros=define_macros,
        extra_compilr_args=[] #extra_compile_args
    )


def get_extensions():

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    sources_cpp = ['./src/convex_cpu.cpp',
                   './src/convex_ext.cpp']
    sources_cu = sources_cpp \
                 + ['./src/convex_cuda.cu']
    include_dirs = [r'./src']
    library_dirs = []
    libraries = []

    ext_modules = [
        CppExtension(
            name="convex_cpu",
            sources=sources_cpp,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            libraries=libraries,
            library_dirs=library_dirs,
        ),
        make_cuda_ext(
            name = 'convex_cuda',
            sources = sources_cu
        ),

    ]

    return ext_modules

setuptools.setup(
    name="convex",
    version=__version__,
    author="xyq",
    author_email='prfans@163.com',
    python_requires=">=3.6",
    install_requires=['numpy'], #, 'opencv347'
    description="description",
    long_description="long_description",
    ext_modules=get_extensions(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    #packages=setuptools.find_packages(),
)
