from transformers import BertTokenizer, BertForSequenceClassification

from base_bert import BertBase


class ItalianBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            tokenizer=BertTokenizer.from_pretrained("dbmdz/bert-base-italian-cased"),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )

    def load_model(
            self,
            num_labels
    ):
        return BertForSequenceClassification.from_pretrained("dbmdz/bert-base-italian-cased",
                                                               num_labels=num_labels,
                                                               output_attentions=False,
                                                               output_hidden_states=False)