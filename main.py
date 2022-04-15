import pandas as pd
import os
import argparse

import consts
from trainer import Trainer
from trainer_config import TrainerConfig


def parse_arguments(args_parser):
    pass


test_or_train = 'test'
data_file_name = f'{test_or_train}_data.tsv'
if __name__ == '__main__':
    data_file_path = os.path.join(consts.answers_students_data, data_file_name)
    df = pd.read_csv(data_file_path, sep='\t')

    trainer_config = TrainerConfig()  # TODO: parse arguments
    trainer = Trainer(df, trainer_config)

    trainer.train()
