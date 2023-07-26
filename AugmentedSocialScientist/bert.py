from transformers import BertTokenizer, BertForSequenceClassification

from AugmentedSocialScientist.base_bert import BertBase


class Bert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )
