import argparse
import numpy as np

from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
import reader
import utils

class Args():
    model_dir = 'infer_model'
    prog_file = 'infer_model/model.pdmodel'
    params_file = 'infer_model/params.pdparams'
    batch_size = 1
    word_dict_path = "./conf/word.dic"
    label_dict_path = "./conf/tag.dic"
    word_rep_dict_path = "./conf/q2b.dic"
    
    
args = Args()
dataset = reader.Dataset(args)

# def text2inds(text_list):
def text2tensor(text):
    
    tensor = PaddleTensor()
    tensor.name = "words"
    tensor.shape = [len(text), 1]
    tensor.dtype = PaddleDType.INT64
    
    UNK = dataset.word2id_dict[u'OOV']
    tensor.data = PaddleBuf([dataset.word2id_dict.get(word, UNK) for word in text])
    
    return tensor

test_data = [u'百度是一家高科技公司', u'中山大学是岭南第一学府']
tensor = [text2tensor(text) for text in test_data]

config = AnalysisConfig(args.model_dir)
config.set_prog_file(args.prog_file)
config.set_params_file(args.params_file)
config.disable_gpu()

# Create PaddlePredictor
predictor = create_paddle_predictor(config)
print(predictor)

outputs = predictor.run([tensor[0]])
print(outputs)