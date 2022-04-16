from datasets import load_dataset
import pandas as pd
import os

import consts
from trainer_config import TrainerConfig
from transformers_pytorch.trainer import TrainerPytorch
from simple_transformers.trainer import TrainerSimpletransformers
from transformers_trainer.trainer import TrainerTrainer

dataset = consts.answers_students_data
test_file_name = f'test_data.tsv'
train_file_name = f'train_data.tsv'
test_path = os.path.join(dataset, test_file_name)
train_path = os.path.join(dataset, train_file_name)


def simple_transformers(trainer_config):
    test_dataframe = pd.read_csv(test_path, sep='\t').astype(str)
    train_dataframe = pd.read_csv(train_path, sep='\t').astype(str)

    trainer = TrainerSimpletransformers(train_dataframe, test_dataframe, trainer_config)
    trainer.train()


def transformers_pytorch(trainer_config):
    train_dataframe = pd.read_csv(train_path, sep='\t').astype(str)

    trainer = TrainerPytorch(train_dataframe, trainer_config)
    trainer.train()


def transformers_trainer(trainer_config):
    pass
    # data = load_dataset('csv', data_files={'train': [train_path], 'test': [test_path]})
    # print(data)
    # train_dataset = dataset['train']
    # val_dataset = dataset['test']

    # trainer = TrainerTrainer(train_dataset, test_dataset, trainer_config)
    # trainer.train()


test_file_name = f'test_data.tsv'
train_file_name = f'test_data.tsv'
if __name__ == '__main__':
    test_path = os.path.join(consts.answers_students_data, test_file_name)
    train_path = os.path.join(consts.answers_students_data, train_file_name)

    config = TrainerConfig()  # TODO: parse arguments

    # simple_transformers(config)
    # transformers_pytorch(config)
    transformers_trainer(config)

