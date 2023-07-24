from transformers import CamembertTokenizer, CamembertForSequenceClassification

from base_bert import BertBase


class CamemBert(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            tokenizer=CamembertTokenizer.from_pretrained('camembert-base'),
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification
        )

    def load_model(
            self,
            num_labels
    ):
        return self.model_sequence_classifier.from_pretrained("camembert-base",
                                                               num_labels=num_labels,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
