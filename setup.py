import os
import multiprocessing
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    os.path.join('inficom', f'pybind.cpp'),
    #    os.path.join('inficom', 'ops', 'attn', f'decode_attn.cu'),
    #    os.path.join('inficom', 'ops', 'norm', f'norm.cu'),
    os.path.join('inficom', 'ops', 'linear', f'gemv.cu'),
    #    os.path.join('inficom', 'ops', 'linear', f'w8.cu'),
    #    os.path.join('inficom', 'ops', 'element', f'residual.cu'),
    #    os.path.join('inficom', 'layers', f'attn_layer.cu'),
    #    os.path.join('inficom', 'layers', f'attn_layer_quant.cu'),
    #    os.path.join('inficom', 'layers', f'ffn_layer.cu'),
    #    os.path.join('inficom', 'layers', f'ffn_layer_quant.cu'),
    #    os.path.join('inficom', 'kernels', 'gemm_s4_f16', f'gemm_s4_f16.cu'),
    #    os.path.join('inficom', 'kernels', 'gemm_s4_f16', f'format.cu'),
]


class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        num_cores = multiprocessing.cpu_count()

        for ext in self.extensions:
            ext.extra_compile_args = ['-j', str(num_cores)]  # 使用-j选项设置线程数
        super().build_extensions()


setup(
    name='inficom',
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            'inficom',
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp', '-lgomp'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--generate-line-info',
                    '-Xptxas=-v',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
