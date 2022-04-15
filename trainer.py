import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

import trainer_config
from ists_dataset import IstsDataset


class Trainer:
    def __init__(self, dataframe: pd.DataFrame, config: trainer_config.TrainerConfig):
        self.config = config

        # TODO:  if {general} else {load_local}
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_type)
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_type)

        dataset = IstsDataset(dataframe, self.tokenizer, self.config.max_input_size, self.config.max_target_size)
        self.loader = DataLoader(dataset)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.learning_rate
        )

    def train(self):
        self.model.train()
        for idx, row in enumerate(self.loader):
            input_ids, input_mask, target_ids = row
            outputs = self.model(  # for training
                input_ids=input_ids,
                attention_mask=input_mask,
                decoder_input_ids=target_ids
            )
            loss = outputs[0]

            if idx % 10 == 0:
                print(loss)

            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            # self.model.eval()
            # generated_ids = self.model.generate(  # for validation
            #     input_ids=input_ids,
            #     attention_mask=input_mask,
            #     max_length=150,
            #     num_beams=2,
            #     repetition_penalty=2.5,
            #     length_penalty=1.0,
            #     early_stopping=True
            # )
            #
            # target = [
            #     self.tokenizer.decode(t,
            #                           skip_special_tokens=True,
            #                           clean_up_tokenization_spaces=True
            #                           )
            #     for t in target_ids]
            # preds = [
            #     self.tokenizer.decode(g,
            #                           skip_special_tokens=True,
            #                           clean_up_tokenization_spaces=True
            #                           )
            #     for g in generated_ids]
            #
            # print(f"actual={target}")
            # print(f"prediction={preds}")
