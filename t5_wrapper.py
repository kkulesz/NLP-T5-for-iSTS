from __future__ import annotations
import torch
from simpletransformers.t5 import T5Model, T5Args

use_cuda = torch.cuda.is_available()


class T5Wrapper:
    def __init__(self, model: T5Model):
        self.model: T5Model = model

    @staticmethod
    def naked(args: T5Args) -> T5Wrapper:
        """
        returns T5Wrapper with model without any training yet
        """
        model = T5Model(
            args.model_type,
            args.model_name,
            args=args,
            use_cuda=use_cuda
        )
        return T5Wrapper(model)

    @staticmethod
    def pretrained(model_dir: str, args: T5Args) -> T5Wrapper:
        """
        returns T5Wrapper with model which was already pretrained and is saved locally
        """
        model = T5Model(
            args.model_type,
            model_dir,
            args=args,
            use_cuda=use_cuda
        )
        return T5Wrapper(model)

    def train(self, data):
        self.model.train_model(
            train_data=data,
            show_running_loss=False
        )

    def predict(self, inputs):
        return self.model.predict(inputs)

