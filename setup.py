import os
import platform
import sys
import tempfile
import warnings
from distutils import ccompiler
from distutils.errors import CompileError, LinkError
from distutils.sysconfig import customize_compiler
from os.path import join

import setuptools
from setuptools import setup, Extension

########################
# TODO: Taken from https://github.com/pavlin-policar/openTSNE/blob/master/setup.py
########################

try:
    from Cython.Distutils.build_ext import new_build_ext as build_ext
    have_cython = True
except ImportError:
    have_cython = False


class get_numpy_include:
    """Helper class to determine the numpy include path

    The purpose of this class is to postpone importing numpy until it is
    actually installed, so that the ``get_include()`` method can be invoked.

    """
    def __str__(self):
        import numpy
        return numpy.get_include()


def get_include_dirs():
    """Get include dirs for the compiler."""
    return (
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Library", "include"),
    )


def get_library_dirs():
    """Get library dirs for the compiler."""
    return (
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "Library", "lib"),
    )


def has_c_library(library, extension=".c"):
    """Check whether a C/C++ library is available on the system to the compiler.

    Parameters
    ----------
    library: str
        The library we want to check for e.g. if we are interested in FFTW3, we
        want to check for `fftw3.h`, so this parameter will be `fftw3`.
    extension: str
        If we want to check for a C library, the extension is `.c`, for C++
        `.cc`, `.cpp` or `.cxx` are accepted.

    Returns
    -------
    bool
        Whether or not the library is available.

    """    
    with tempfile.TemporaryDirectory(dir=".") as directory:
        name = join(directory, "%s%s" % (library, extension))
        print(name)
        with open(name, "w") as f:
            f.write("#include <%s.h>\n" % library)
            f.write("int main() {}\n")

        # Get a compiler instance
        compiler = ccompiler.new_compiler()
        # Configure compiler to do all the platform specific things
        customize_compiler(compiler)
        # Add conda include dirs
        for inc_dir in get_include_dirs():
            compiler.add_include_dir(inc_dir)
        assert isinstance(compiler, ccompiler.CCompiler)

        try:
            # Try to compile the file using the C compiler
            compiler.link_executable(compiler.compile([name]), name)
            return True
        except (CompileError, LinkError):
            return False


class CythonBuildExt(build_ext):
    def build_extensions(self):
        if not have_cython:
            raise RuntimeError("Missing build dependency: Cython")

        extra_compile_args = []
        extra_link_args = []

        # Optimization compiler/linker flags are added appropriately
        compiler = self.compiler.compiler_type
        if compiler == "unix":
            extra_compile_args += ["-O3"]
        elif compiler == "msvc":
            extra_compile_args += ["/Ox", "/fp:fast"]

        if compiler == "unix":
            # https://stackoverflow.com/questions/22931147/stdisinf-does-not-work-with-ffast-math-how-to-check-for-infinity
            extra_compile_args += [
                "-ffast-math",
                "-fno-finite-math-only",  # we use infinity
                "-fno-associative-math",
            ]

        # Set minimum deployment version for MacOS
        if compiler == "unix" and platform.system() == "Darwin":
            extra_compile_args += ["-mmacosx-version-min=10.12"]
            extra_link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.12"]

        # We don't want the compiler to optimize for system architecture if
        # we're building packages to be distributed by conda-forge, but if the
        # package is being built locally, this is desired
        if not ("AZURE_BUILD" in os.environ or "CONDA_BUILD" in os.environ):
            if platform.machine() == "ppc64le":
                extra_compile_args += ["-mcpu=native"]
            if platform.machine() == "x86_64":
                extra_compile_args += ["-march=native"]

        # We will disable openmp flags if the compiler doesn"t support it. This
        # is only really an issue with OSX clang
        if has_c_library("omp"):
            print("Found openmp. Compiling with openmp flags...")
            if platform.system() == "Darwin" and compiler == "unix":
                extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                extra_link_args += ["-lomp"]
            elif compiler == "unix":
                extra_compile_args += ["-fopenmp"]
                extra_link_args += ["-fopenmp"]
            elif compiler == "msvc":
                extra_compile_args += ["/openmp"]
                extra_link_args += ["/openmp"]
        else:
            warnings.warn(
                "You appear to be using a compiler which does not support "
                "openMP, meaning that the library will not be able to run on "
                "multiple cores. Please install/enable openMP to use multiple "
                "cores."
            )

        for extension in self.extensions:
            extension.extra_compile_args += extra_compile_args
            extension.extra_link_args += extra_link_args

        # Add numpy and system include directories
        for extension in self.extensions:
            extension.include_dirs.extend(get_include_dirs())
            extension.include_dirs.append(get_numpy_include())

        # Add numpy and system include directories
        for extension in self.extensions:
            extension.library_dirs.extend(get_library_dirs())

        super().build_extensions()


extensions = [
    Extension("hyperbolicTSNE.hyperbolic_barnes_hut.tsne_utils",
              sources=["hyperbolicTSNE/hyperbolic_barnes_hut/tsne_utils.pyx"],
              language="c++"),
    Extension("hyperbolicTSNE.hyperbolic_barnes_hut.tsne",
              sources=["hyperbolicTSNE/hyperbolic_barnes_hut/tsne.pyx"],
              language="c++"),
    Extension("hyperbolicTSNE.hyperbolic_barnes_hut.tree",
              sources=["hyperbolicTSNE/hyperbolic_barnes_hut/tree.pyx"],
              language="c++"),
]


def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="hyperbolic-tsne",
    description="Hyperbolic implementation of t-SNE",
    long_description=readme(),
    version="0.1.0",
    # license="BSD-3-Clause",
    # author="Hunter van Geffen",
    # author_email="huntervangeffen@gmail.com",
    url="https://github.com/chadepl/hyperbolic-tsne",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        # "License :: OSI Approved",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    packages=setuptools.find_packages(include=["hyperbolicTSNE", "hyperbolicTSNE.*"]),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.21",
        "scikit-learn>=0.20",
        "scipy",
        "tqdm"
    ],
    setup_requires=[
        "cython",
    ],
    extras_require={
        "plot": [
            "pandas",
            "matplotlib",
            "seaborn",
        ],
        "anndata": "anndata",
        "hnsw": "hnswlib~=0.4.0",
        "pynndescent": "pynndescent~=0.5.0",
    },
    ext_modules=extensions,
    cmdclass={"build_ext": CythonBuildExt},
)
