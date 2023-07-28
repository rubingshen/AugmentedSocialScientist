from transformers import BertTokenizer, BertForSequenceClassification, \
                         CamembertTokenizer, CamembertForSequenceClassification, \
                         XLMRobertaForSequenceClassification, XLMRobertaTokenizer
                         

from AugmentedSocialScientist.bert_base import BertBase


class Bert(BertBase):
    def __init__(
            self,
            model_name='bert-base-uncased',
            device=None
    ):
        super().__init__(
            model_name=model_name, 
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class ArabicBert(BertBase):
    def __init__(
            self,
            model_name="asafaya/bert-base-arabic",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class Camembert(BertBase):
    def __init__(
            self,
            model_name='camembert-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification
        )


class ChineseBert(BertBase):
    def __init__(
            self,
            model_name="bert-base-chinese",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer.from_pretrained("bert-base-chinese"),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class GermanBert(BertBase):
    def __init__(
            self,
            model_name="dbmdz/bert-base-german-uncased",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class HindiBert(BertBase):
    def __init__(
            self,
            model_name="monsoon-nlp/hindi-bert",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class ItalianBert(BertBase):
    def __init__(
            self,
            model_name="dbmdz/bert-base-italian-cased",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class PortugueseBert(BertBase):
    def __init__(
            self,
            model_name='neuralmind/bert-base-portuguese-cased',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class RussianBert(BertBase):
    def __init__(
            self,
            model_name="DeepPavlov/rubert-base-cased",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class SpanishBert(BertBase):
    def __init__(
            self,
            model_name="dccuchile/bert-base-spanish-wwm-uncased",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class SwedishBert(BertBase):
    def __init__(
            self,
            model_name='KB/bert-base-swedish-cased',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class XLMRoberta(BertBase):
    def __init__(
            self,
            model_name='xlm-roberta-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=XLMRobertaTokenizer,
            device=device,
            model_sequence_classifier=XLMRobertaForSequenceClassification
        )