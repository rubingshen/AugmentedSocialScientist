from setuptools import setup

environment = [
    'scikit-learn',
    'tqdm',
    'pandas',
    'numpy',
    'scipy',
    'nltk',
    'keras',
    'tensorflow',
    'torch',
    'transformers',
    'sentencepiece',
]

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(name='AugmentedSocialScientist',
      author='Rubing Shen',
      license='MIT',
      author_email='shenrubing1996@gmail.com',
      url='https://github.com/rubingshen/AugmentedSocialScientist',
      download_url='https://github.com/rubingshen/AugmentedSocialScientist/archive/refs/tags/v1.1.0.tar.gz',
      version='1.1.0',
      description='A Simple Package to Train Bert-Like Model for Text Classification',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['AugmentedSocialScientist'],
      zip_safe=False,
      install_requires=environment)
