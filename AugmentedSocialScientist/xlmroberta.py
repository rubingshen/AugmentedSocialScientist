from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

from base_bert import BertBase


class XMLRoBerta(BertBase):
    def __init__(
            self,
            device=None
    ):
        super().__init__(
            tokenizer=XLMRobertaTokenizer.from_pretrained('xlm-roberta-base'),
            device=device,
            model_sequence_classifier=XLMRobertaForSequenceClassification
        )

    def load_model(
            self,
            num_labels
    ):
        return XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                               num_labels = num_labels,
                                                               output_attentions = False,
                                                               output_hidden_states = False)
