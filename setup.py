from setuptools import find_packages, setup

import os
import torch
from os import path as osp
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)



if __name__ == '__main__':
    # add_mim_extention()
    setup(
        name='OpenOccupancy',
        version='0.0',
        description=("OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception"),
        author='OpenOccupancy Contributors',
        author_email='wangxiaofeng2020@ia.ac.cn',
        keywords='Occupancy Perception',
        packages=find_packages(),
        include_package_data=True,
        package_data={'projects.occ_plugin.ops': ['*/*.so']},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        license="Apache License 2.0",

        ext_modules=[
            make_cuda_ext(
                name="occ_pool_ext",
                module="projects.occ_plugin.ops.occ_pooling",
                sources=[
                    "src/occ_pool.cpp",
                    "src/occ_pool_cuda.cu",
                ]),
        ],
        cmdclass={'build_ext': BuildExtension})