from setuptools import setup

environment = [
    'scikit-learn',
    'tqdm',
    'pandas',
    'numpy',
    'scipy',
    'nltk',
    'keras==2.8',
    'tensorflow==2.8;sys_platform!="darwin"',
    'tensorflow-macos==2.8;sys_platform=="darwin"',
    # cf. https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly
    'protobuf==3.20.*',
    'torch',
    'transformers',
    'sentencepiece',
]

setup(name='AugmentedSocialScientist',
      author='Rubing Shen',
      license='MIT',
      author_email='rubing.shen@sciencespo.fr',
      version='0.1',
      description='wrapper for bert-family models training',
      packages=['AugmentedSocialScientist'],
      zip_safe=False,
      install_requires=environment)
