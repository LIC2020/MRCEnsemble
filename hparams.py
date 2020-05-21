import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # model version
    parser.add_argument('--version', default='v1', type=str)
    parser.add_argument('--division', default='dev', type=str)

    ## agg setting
    parser.add_argument('--use_weight', default=False, action='store_true',
                        help="If use f1 as the weight")
    parser.add_argument('--use_rule', default=False, action='store_true',
                        help="If use f1 as the weight")
    ## for V1
    parser.add_argument('--use_logit', default=False, action='store_true',
                        help="If use sigmoid(satrt_logit + end_logit) as the score of each answer")
    
    ## for V2
    parser.add_argument('--use_decay', default=False, action='store_true',
                        help="If use decay")
    parser.add_argument('--decay_rate', default=0.9, type=float)
    parser.add_argument('--weight_constant', default=72, type=float)
    

    # other
    parser.add_argument('--topN', default=None, type=int,
                        help="topN answers of each nbest result will be used")
    parser.add_argument('--agg_method', default="sum", type=str,
                        help="Indicate how to aggregate the answer scores, use sum or avg")
   
    # filepath
    parser.add_argument('--nbest_dir', default="./results/", type=str,
                        help="nbest_file_data")
    parser.add_argument('--question_path', default=None, type=str,
                        help="If use_rule=True, you should feed the question file path")
    parser.add_argument('--output_path', default=None, type=str,
                        help="ensemble results path")
   


