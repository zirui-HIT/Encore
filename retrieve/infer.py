import sys
from tqdm import tqdm
from utils import *
import argparse
from config import parameters as conf
from torch import nn
import torch
import transformers

sys.path.append(".")

if conf.pretrained_model == "bert":
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "roberta":
    from transformers import RobertaTokenizer
    from transformers import RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)


def generate(data_ori, data, model, mode='valid'):
    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, shuffle=False)

    k = 0
    all_logits = []
    all_filename_id = []
    all_ind = []
    with torch.no_grad():
        for x in tqdm(data_iterator):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            label = x['label']
            filename_id = x["filename_id"]
            ind = x["ind"]

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids]:
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)

            logits = model(True, input_ids, input_mask,
                           segment_ids, device=conf.device)

            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)

    output_prediction_file = args.output_path

    if mode == "valid":
        print_res = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.valid_file, topn=conf.topn)
    elif mode == "test":
        print_res = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.test_file, topn=conf.topn)
    else:
        # private data mode
        print_res = retrieve_evaluate_private(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.test_file, topn=conf.topn)

    write_log(args.log_path, print_res)
    print(print_res)
    return


def generate_test():
    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,)

    model = nn.DataParallel(model)
    model.to(conf.device)
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    generate(test_data, test_features, model, mode='test')


if __name__ == '__main__':
    from retrieve.model import Bert_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--part", "-p", choices=['train', 'dev', 'test'])
    parser.add_argument("--save_path", "-sp", type=str)
    parser.add_argument("--log_path", "-lp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    args = parser.parse_args()
    transformers.set_seed(42)

    conf.train_file = f"{args.data_path}/train.json"
    conf.valid_file = f"{args.data_path}/dev.json"
    conf.test_file = f"{args.data_path}/{args.part}.json"

    op_list = []
    const_list = []
    reserved_token_size = len(op_list) + len(const_list)

    test_data, test_examples, op_list, const_list = \
        read_examples(input_path=conf.test_file, tokenizer=tokenizer,
                      op_list=op_list, const_list=const_list, log_file=args.log_path, mode='test')

    kwargs = {"examples": test_examples,
              "tokenizer": tokenizer,
              "option": conf.option,
              "is_training": False,
              "max_seq_length": conf.max_seq_length,
              }

    kwargs["examples"] = test_examples
    test_features = convert_examples_to_features(**kwargs)

    generate_test()
