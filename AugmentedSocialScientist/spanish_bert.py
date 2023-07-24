from transformers import BertTokenizer, BertForSequenceClassification

from base_bert import BertBase


class SpanishBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            tokenizer=BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased"),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )

    def load_model(
            self,
            num_labels
    ):
        return BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased",
                                                               num_labels=num_labels,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
