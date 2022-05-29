import torch
import numpy as np
import os
import random
import warnings
from optparse import OptionParser

from config import Config


def seed_torch(seed=2137):
    #  taken from: https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def turn_off_stupid_warnings():
    # ugly, but simpletransfomers.T5 throws some stupid
    # future warnings if everything is done the way
    # the official tutorial says: https://simpletransformers.ai/docs/t5-model/
    warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_environment():
    turn_off_stupid_warnings()
    seed_torch()


def parse_arguments() -> Config:
    # TODO dorobić obsługę pozostałych optionów
    usage = "usage: %prog [options]\n" \
            "Params:\t-t (model type), -n (model name), \n" \
            "\t-r (learning rate), -o (model output dir), \n" \
            "\t-v (variant), -d(dataset) \n"
    parser = OptionParser(usage=usage)

    parser.add_option("-t", "--model_type", type="string", dest="model_type", default="t5")
    parser.add_option("-n", "--model_name", type="string", dest="model_name", default="t5-small")
    parser.add_option("-r", "--learning_rate", type="float", dest="learning_rate", default=1e-4)
    parser.add_option("-o", "--model_output_dir", type="string", dest="model_output_dir", default="model_storage")
    # parser.add_option("-v", "--variant", type="string", dest="variant", default="both")
    # parser.add_option("-d", "--dataset", type="string", dest="dataset", default="/data/answers-students")
    # parser.add_option("-e", "--epochs", type="int", dest="epochs", default=1)
    (options, args) = parser.parse_args()

    return Config(model_type=options.model_type,
                  model_name=options.model_name,
                  learning_rate=options.learning_rate,
                  model_output_dir=options.model_output_dir)
