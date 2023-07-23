# The Augmented Social Scientist


This package allows to simply train BERT-like models for text classifications. 

It comes with our article "[The Augmented Social Scientist: Using Sequential Transfer Learning to Annotate Millions of Texts with Human-Level Accuracy](https://journals.sagepub.com/doi/abs/10.1177/00491241221134526)" published on *Sociological Methods & Research* by [Salomé Do](https://sally14.github.io), [Étienne Ollion](https://ollion.cnrs.fr/english/) and [Rubing Shen](https://rubingshen.github.io). 



## To install the package

- Use pip
```
pip install AugmentedSocialScientist
```

- Or from source
```
git clone https://github.com/rubingshen/AugmentedSocialScientist.git  
pip install ./AugmentedSocialScientist
```

## Import BERT model
```python
from AugmentedSocialScientist import bert
```

The module `bert` contains 3 main functions:
- `bert.encode()` to preprocess the data;
- `bert.run_training()` to train, validate and save a model;
- `bert.predict_with_model()`  to make predictions with a saved model.

## Tutorial
Check [here](https://colab.research.google.com/drive/132_oDik-SOWve31tZ8D1VOx1Sj_Cyzn7?usp=sharing) for a Google Colab tutorial.

## Languages supported

BERT is a pre-trained language model for the English language. The package also contains models for other languages:
- `camembert` for French;
- `arabic_bert` for Arabic;
- `chinese_bert` for Chinese;
- `german_bert` for German;
- `hindi_bert` for Hindi;
- `italian_bert` for Italian;
- `portuguese_bert` for Portuguese;
- `russian_bert` for Russian;
- `spanish_bert` for Spanish;
- `swedish_bert` for Swedish;
- `xlmroberta` which is a multi-lingual model supporting 100 languages.


To use them, simply import the corresponding model and replace `bert` with the name of the imported model.

For example, to use the French language model `camembert`:
 1. Import the model `camembert`:
```python
from AugmentedSocialScientist import camembert
```
 2. Then use the functions `camembert.encode()`, `camembert.run_training()`, `camembert.predict_with_model()`


