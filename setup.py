from setuptools import setup

environment = [
    'scikit-learn',
    'tqdm',
    'numpy',
    'scipy',
    'torch',
    'transformers',
    'sentencepiece'
]

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(name='AugmentedSocialScientist',
      author='Rubing Shen',
      license='MIT',
      author_email='shenrubing1996@gmail.com',
      url='https://github.com/rubingshen/AugmentedSocialScientist',
      download_url='https://github.com/rubingshen/AugmentedSocialScientist/archive/refs/tags/v2.0.1.tar.gz',
      version='2.0.1',
      description='A Package to Easily Train Bert-Like Models for Text Classification',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['AugmentedSocialScientist'],
      zip_safe=False,
      install_requires=environment)
