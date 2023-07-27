from transformers import BertTokenizer, BertForSequenceClassification

from AugmentedSocialScientist.base_bert import BertBase


class CustomBert(BertBase):
    def __init__(
            self,
            custom_model,
            device=None
    ):
        super().__init__(
            tokenizer=BertTokenizer.from_pretrained(custom_model),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )

        """
        Parameter
        ---------
        custom_model: string, model name from hugging face models: https://huggingface.co/models
        """
        
        self.custom_model = custom_model

    def load_model(
            self,
            num_labels
    ):
        return BertForSequenceClassification.from_pretrained(self.custom_model,
                                                      num_labels=num_labels,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
