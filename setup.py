from setuptools import setup

environment = [
    'scikit-learn',
    'tqdm',
    'numpy',
    'scipy',
    'torch',
    'transformers',
    'sentencepiece',
]

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(name='augmentedsocialscientist',
      author='Rubing Shen',
      license='MIT',
      author_email='shenrubing1996@gmail.com',
      url='https://github.com/rubingshen/augmentedsocialscientist',
      download_url='https://github.com/rubingshen/augmentedsocialscientist/archive/refs/tags/v3.0.0.tar.gz',
      version='3.0.0',
      description='A Package to Easily Train Bert-Like Models for Text Classification',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['augmentedsocialscientist'],
      zip_safe=False,
      install_requires=environment)
