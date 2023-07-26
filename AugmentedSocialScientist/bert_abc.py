from abc import ABC, abstractmethod


class BertABC(ABC):
    """
    Abstract class defining that Bert implementations can inherit from.
    """

    @abstractmethod
    def __init__(
            self,
            tokenizer,
            device=None
    ):
        pass

    @abstractmethod
    def encode(
            self,
            sentences,
            labels=None,
            batch_size=32,
            progress_bar=True
    ):
        pass

    @abstractmethod
    def run_training(
            self,
            train_dataloader,
            test_dataloader,
            n_epochs=3,
            lr=5e-5,
            random_state=2018,
            save_model_as=None
    ):
        pass

    @abstractmethod
    def predict(
            self,
            dataloader,
            model,
            proba=True,
            progress_bar=True
    ):
        pass

    @abstractmethod
    def predict_with_model(
            self,
            dataloader,
            model_path,
            proba=True,
            progress_bar=True
    ):
        pass

    @abstractmethod
    def format_time(
            self,
            elapsed
    ):
        pass
