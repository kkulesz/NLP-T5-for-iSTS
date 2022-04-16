import pandas as pd

import trainer_config

from simple_transformers.data_processing import for_type, for_score
from simpletransformers.t5 import T5Model

model_args = {  # TODO: from config
    "max_seq_length": 196,
    "train_batch_size": 16,
    "eval_batch_size": 64,
    "num_train_epochs": 1,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "fp16": False,
    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "reprocess_input_data": True,
    "overwrite_output_dir": True
    # "wandb_project": "T5 mixed tasks - Binary, Multi-Label, Regression",
}


class TrainerSimpletransformers:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, config: trainer_config.TrainerConfig):
        self.train_score_df = for_score(train_df)
        self.train_type_df = for_type(train_df)

        self.test_score_df = for_score(test_df)
        self.test_type_df = for_type(test_df)

        self.config = config
        self.model = T5Model("t5", self.config.model_type, use_cuda=False, args=model_args)

    def train(self):
        self.model.train_model(
            self.train_score_df,
            eval_data=self.test_score_df,
            output_dir="output"
        )
        # TODO save and load model somehow
