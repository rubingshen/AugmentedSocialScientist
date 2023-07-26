from transformers import BertTokenizer, BertForSequenceClassification

from AugmentedSocialScientist.base_bert import BertBase


class ArabicBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            tokenizer=BertTokenizer.from_pretrained("asafaya/bert-base-arabic"),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )

    def load_model(
            self,
            num_labels
    ):
        return BertForSequenceClassification.from_pretrained("asafaya/bert-base-arabic",
                                                      num_labels=num_labels,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
