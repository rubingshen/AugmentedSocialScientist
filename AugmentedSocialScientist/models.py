from transformers import BertTokenizer, BertForSequenceClassification, \
                         CamembertTokenizer, CamembertForSequenceClassification, \
                         XLMRobertaForSequenceClassification, XLMRobertaTokenizer
                         

from AugmentedSocialScientist.bert_base import BertBase


class Bert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name='bert-base-uncased', 
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class ArabicBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name="asafaya/bert-base-arabic",
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class Camembert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name='camembert-base',
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification
        )


class ChineseBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name="bert-base-chinese",
            tokenizer=BertTokenizer.from_pretrained("bert-base-chinese"),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class GermanBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name="dbmdz/bert-base-german-uncased",
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class HindiBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name="monsoon-nlp/hindi-bert",
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class ItalianBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name="dbmdz/bert-base-italian-cased",
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class PortugueseBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name='neuralmind/bert-base-portuguese-cased',
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class RussianBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name="DeepPavlov/rubert-base-cased",
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class SpanishBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name="dccuchile/bert-base-spanish-wwm-uncased",
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class SwedishBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name='KB/bert-base-swedish-cased',
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class XLMRoberta(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            model_name='xlm-roberta-base',
            tokenizer=XLMRobertaTokenizer,
            device=device,
            model_sequence_classifier=XLMRobertaForSequenceClassification
        )