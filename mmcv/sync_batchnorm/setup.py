from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='syncbn',
    ext_modules=[
        CUDAExtension(name='syncbn',
                      sources=['csrc/syncbn.cpp',
                               'csrc/welford.cu'])
    ],
    cmdclass={'build_ext': BuildExtension})
