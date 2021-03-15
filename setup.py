from setuptools import setup, find_packages

setup(
    name='awesome_align',
    install_requires=[
        'tokenizers>=0.5.2',
        'torch>=1.1.0',
        'tqdm',
        'numpy',
        'boto3',
        'filelock'
    ],
    version='0.1.2',
    author='NeuLab',
    author_email='zdou0830@gmail.com',
    license='BSD 3-Clause',
    description='An awesome word alignment tool',
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "awesome-align=awesome_align.run_align:main",
            "awesome-train=awesome_align.run_train:main",
        ],
    },
    url='https://github.com/neulab/awesome-align',
)
