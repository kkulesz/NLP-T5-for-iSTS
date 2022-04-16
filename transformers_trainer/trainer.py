import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers import TrainingArguments, Trainer

from datasets import load_dataset

import trainer_config


class TrainerTrainer:
    def __init__(self, train_dataset, test_dataset, config: trainer_config.TrainerConfig):
        self.config = config
        self.training_args = TrainingArguments(
            output_dir="trainer_output"
        )

        self.model = T5Model.from_pretrained(self.config.model_type)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            # compute_metrics=compute_metrics,
        )

    def train(self):
        pass
