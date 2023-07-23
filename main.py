import argparse
import os.path as osp

from GSL.experiment import Experiment

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='the name of graph dataset')
    parser.add_argument('--model', type=str, default='GRCN',
                        help='the name of GSL model')
    parser.add_argument('--ntrail', type=int, default=1,
                        help='repetition count of experiments')
    parser.add_argument('--use_knn', action='store_true',
                        help='whether to use knn graph instead of the original structure')
    parser.add_argument('--k', type=int, default=5, help='the number of nearest neighbors')
    parser.add_argument('--drop_rate', type=float, default=0.,
                        help='the probability of randomly drop edges')
    parser.add_argument('--add_rate', type=float, default=0.,
                        help='the probability of randomly add edges')
    parser.add_argument('--mask_feat_rate', type=float, default=0.,
                        help='the probability of randomly mask features')
    parser.add_argument('--use_mettack', action='store_true',
                        help='whether to use the structure after being attacked by mettack')
    parser.add_argument('--ptb_rate', type=float, default=0.,
                        help='the perturbation rate')
    parser.add_argument('--metric', type=str, default='acc',
                        help='the evaluation metric')
    parser.add_argument('--sparse', action='store_true',
                        help='whether to use sparse version')
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='the selected GPU number')
    parser.add_argument('--learner', type=str, default='MLP', choices=['FP', 'MLP'],
                        help='the learner of SLAPS')
    args = parser.parse_args()

    data_path = osp.join(osp.expanduser('~'), 'datasets')
    config_path = './configs/{}_config.yaml'.format(args.model.lower())
    if args.model.lower() == 'slaps':
        config_path = './configs/{}_{}_config.yaml'.format(args.model.lower(), args.learner.lower())

    exp = Experiment(model_name=args.model, dataset=args.dataset, ntrail=args.ntrail,
                     data_path=data_path, config_path=config_path, metric=args.metric, sparse=args.sparse,
                     use_knn=args.use_knn, k=args.k, drop_rate=args.drop_rate, gpu_num=args.gpu_num,
                     add_rate=args.add_rate, use_mettack=args.use_mettack, ptb_rate=args.ptb_rate, mask_feat_rate=args.mask_feat_rate)
    exp.run()
    # exp.hp_search("hyperparams/idgl_hyper.yaml")