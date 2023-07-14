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

setup(name='AugmentedSocialScientist',
      author='Rubing Shen',
      license='MIT',
      author_email='shenrubing1996@gmail.com',
      url='https://github.com/rubingshen/AugmentedSocialScientist',
      download_url='https://github.com/rubingshen/AugmentedSocialScientist/archive/refs/tags/v1.0.0.tar.gz',
      version='1.0.0',
      description='a simple package to train bert-like model for text classification',
      packages=['AugmentedSocialScientist'],
      zip_safe=False,
      install_requires=environment)
