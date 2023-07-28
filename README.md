# The Augmented Social Scientist

**New release v2**

This package makes it extremely easy to train BERT-like models for text classifications. 

For more information on the method and for some use cases from social sciences, see "[The Augmented Social Scientist: Using Sequential Transfer Learning to Annotate Millions of Texts with Human-Level Accuracy](https://journals.sagepub.com/doi/abs/10.1177/00491241221134526)" published on *Sociological Methods & Research* by [Salomé Do](https://sally14.github.io), [Étienne Ollion](https://ollion.cnrs.fr/english/) and [Rubing Shen](https://rubingshen.github.io). 



## To install the package

- Use pip
```
pip install AugmentedSocialScientist
```

- Or, from source
```
git clone https://github.com/rubingshen/AugmentedSocialScientist.git  
pip install ./AugmentedSocialScientist
```

## To train a BERT model

### Import 

```python
from AugmentedSocialScientist.models import Bert

bert = Bert()  #instanciation
```

### Functions 

The class `Bert` contains 3 main methods:
- `encode()` to preprocess the data;
- `run_training()` to train, validate and save a model;
- `predict_with_model()`  to make predictions with a saved model.

### Example

```python
import pandas as pd
import numpy as np

from AugmentedSocialScientist.models import Bert

bert = Bert() #instanciation


# 1. Training 
## Load training and validation data
train_data = pd.read_csv('https://raw.githubusercontent.com/rubingshen/augmented_tutorial/main/clickbait/clickbait_train.csv')
test_data = pd.read_csv('https://raw.githubusercontent.com/rubingshen/augmented_tutorial/main/clickbait/clickbait_test.csv')

## Preprocess the training and validation data
train_loader = bert.encode(
    train.text.values,      #list of texts
    train.label.values      #list of labels
    )    
test_loader = bert.encode(
    test.text.values,       #list of texts
    test.label.values       #list of labels
    )      

## Train, validate and save a model
scores = bert.run_training(
    train_loader,             #training dataloader
    test_loader,              #test dataloader
    lr=5e-5,                  #learning rate
    n_epochs=3,               #number of epochs
    random_state=42,          #random state (for replicability)
    save_model_as='clickbait' #name of model to save as
)
# this trained model will be saved at ./models/clickbait
# the output "scores" contains precision, recall, f1-score and support for each classification category, assessed against the provided test set

# 2. Prediction on unlabeled data

## Load prediction data
pred_data = pd.read_csv('https://raw.githubusercontent.com/rubingshen/augmented_tutorial/main/clickbait/clickbait_pred.csv')

## Preprocess the prediction data
pred_loader = bert.encode(pred_data.text.values) #input a list of unlabeld texts

## Prediction with the saved trained model
pred = bert.predict_with_model(
    pred_loader, 
    model_path='./models/clickbait'
    )
# the output "pred" is a ndarray containing probabilities for each text (row) of belonging to each category (column)

## Compute the predicted label as the one with the highest probability

pred_data['pred_label'] = np.argmax(pred, axis=1)
pred_data['pred_proba'] = np.max(pred, axis=1)
```



### Tutorial
Check [here](https://colab.research.google.com/drive/132_oDik-SOWve31tZ8D1VOx1Sj_Cyzn7?usp=sharing) for an interactive tutorial on Google Colab.

## Languages supported

`Bert` is a pre-trained language model for the English language. The module `AugmentedSocialScientist.models` also contains models for other languages:

- `ArabicBert` for Arabic;
- `Camembert` for French;
- `ChineseBert` for Chinese;
- `GermanBert` for German;
- `HindiBert` for Hindi;
- `ItalianBert` for Italian;
- `PortugueseBert` for Portuguese;
- `RussianBert` for Russian;
- `SpanishBert` for Spanish;
- `SwedishBert` for Swedish;
- `XLMRoberta` which is a multi-lingual model supporting 100 languages.


To use them, just import the corresponding model and instanciate it as in the previous example.

For example, to use the French language model `Camembert`:
```python
from AugmentedSocialScientist.models import Camembert

bert = Camembert()  #instanciation
```
You can then use the functions `bert.encode()`, `bert.run_training()`, `bert.predict_with_model()` as in the previous example.

## To use a custom model from Hugging Face

The package also allows you to use other BERT-like models from [Hugging Face](https://huggingface.co/models), by changing the argument `model_name` to the desired model name when instanciating the class `Bert`. 

For example, to use the Danish BERT model [DJSammy/bert-base-danish-uncased_BotXO-ai](https://huggingface.co/DJSammy/bert-base-danish-uncased_BotXO-ai) from Hugging Face: 

```python
from AugmentedSocialScientist.models import Bert

bert = Bert(model_name="DJSammy/bert-base-danish-uncased_BotXO-ai")
``````

## To use a custom `torch.Device`
By default, the package automatically detects the presence of a GPU and uses it to accelerate computation. You can also set your own device, by providing a `torch.Device` object to the parameter `device` when instanciating `Bert`.

```python
from AugmentedSocialScientist.models import Bert

bert = Bert(device=...)  #set your own device
```
