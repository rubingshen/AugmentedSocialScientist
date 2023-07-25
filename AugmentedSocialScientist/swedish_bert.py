from transformers import BertTokenizer, BertForSequenceClassification

from AugmentedSocialScientist.base_bert import BertBase


class SwedishBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            tokenizer=BertTokenizer.from_pretrained('KB/bert-base-swedish-cased'),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )

    def load_model(
            self,
            num_labels
    ):
        return BertForSequenceClassification.from_pretrained('KB/bert-base-swedish-cased',
                                                               num_labels=num_labels,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
