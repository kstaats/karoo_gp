import setuptools
from karoo_gp import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="karoo_gp",
    version=__version__,
    author="Kai Staats",
    author_email="github@overthesun.com",
    description="Use Genetic Programming for Classification and Symbolic Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kstaats/karoo_gp",
    project_urls={
        'Documentation': 'https://github.com/kstaats/karoo_gp/blob/master/Karoo_GP_User_Guide.pdf',
        'Source': 'https://github.com/kstaats/karoo_gp',
        'Tracker': 'https://github.com/kstaats/karoo_gp/issues',
    },
    license="MIT",
    packages=["karoo_gp"],
    scripts=["karoo-gp.py"],
    package_data={
        'karoo_gp': ['files/*'],
    },
    install_requires=[
        'numpy',
        'sklearn',
        'sympy',
        'tensorflow',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
    ],
    python_requires='>=3',
)
