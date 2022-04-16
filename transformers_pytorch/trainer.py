import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model

import trainer_config
from transformers_pytorch.ists_dataset import IstsDataset


# https://www.kaggle.com/code/prithvijaunjale/t5-multi-label-classification/notebook
# https://huggingface.co/docs/transformers/training
class TrainerPytorch:
    def __init__(self, dataframe: pd.DataFrame, config: trainer_config.TrainerConfig):
        self.config = config

        self.model = T5Model.from_pretrained(self.config.model_type)
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
            score_tokenized, type_tokenized = row

            score_input_ids, score_input_mask, score_target_ids = score_tokenized
            type_input_ids, type_input_mask, type_target_ids = type_tokenized

            outputs = self.model(
                input_ids=score_input_ids,
                attention_mask=score_input_mask,
                decoder_input_ids=score_target_ids
            )
            # loss = outputs.loss
            #
            # if idx % 10 == 0:
            #     print(loss)
            #
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()


        #     self.model.eval()
        #     generated_ids = self.model.generate(
        #         input_ids=score_input_ids,
        #         attention_mask=score_input_mask,
        #         max_length=150,
        #         num_beams=2,
        #         repetition_penalty=2.5,
        #         length_penalty=1.0,
        #         early_stopping=True
        #     )
        #
        #     target = [
        #         self.tokenizer.decode(t,
        #                               skip_special_tokens=True,
        #                               clean_up_tokenization_spaces=True
        #                               )
        #         for t in score_target_ids]
        #     preds = [
        #         self.tokenizer.decode(g,
        #                               skip_special_tokens=True,
        #                               clean_up_tokenization_spaces=True
        #                               )
        #         for g in generated_ids]
        #
        #     print(f"actual={target}")
        #     print(f"prediction={preds}")
