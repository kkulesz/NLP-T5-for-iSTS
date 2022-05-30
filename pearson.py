# encoding: utf-8

from scipy.stats import pearsonr
import pandas as pd

def pearson(datatsv, outcsv):
    col_list_data = ["x1","x2",	"y_type", "y_score"]
    col_list_out = ["type_predictions", "score_predictions"]

    data = pd.read_csv(datatsv, usecols=col_list_data, delimiter='\t')
    out = pd.read_csv(outcsv, usecols=col_list_out, delimiter=',')

    data_type = data['y_type']
    out_type = out['type_predictions']
    data_score = data['y_score']
    out_score = out['score_predictions']

    print("Pearson for score: {}".format(pearsonr(data_score, out_score)[0]))
    print("Pearson for type: {}".format(pearsonr(data_type, out_type)[0]))

if __name__ == '__main__':
    pearson("data/headlines/test_data.tsv", "out.csv")