from CharTrajectories.run import main
import ml_collections
import yaml
import numpy as np
from CharTrajectories.utils import load_config
import logging
logging.basicConfig(filename='CharTrajectories_run_all',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
baseline_models = ['LSTM', 'signature']

params = ['SO', 'Sp']
drop_rate_list = [0.3, 0.5, 0.7]
runs = 5

print('Run all experiments on Character Trajectories dataset')
test_accs = []
# run LSTM_DEV model

for param in params:
    for drop_rate in drop_rate_list:
        for i in range(runs):
            config = load_config(
                'CharTrajectories/configs/train_lstm_dev.yaml')
            print(
                'Running LSTM+DEV({}) on drop rate ={}, run {}/5 '.format(param, drop_rate, i+1))
            config.param = param
            config.drop_rate = drop_rate
            test_acc = main(config=config)
            test_accs.append(test_acc)
        print('LSTM+DEV({}) on CT dataset with drop rate ={},has mean test accuracy {} with std {}'.format(param, drop_rate,
                                                                                                           np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
        logging.info('LSTM+DEV({}) on CT dataset with drop rate ={},has mean test accuracy {} with std {}'.format(param, drop_rate,
                                                                                                                  np.mean(np.array(test_accs)), np.std(np.array(test_accs))))

# run DEV model
test_accs = []
for drop_rate in drop_rate_list:
    for i in range(runs):
        config = load_config(
            'CharTrajectories/configs/train_dev.yaml')
        print('Running DEV({}) on drop rate ={}, run {}/5 '.format('SO', drop_rate, i+1))
        config.drop_rate = drop_rate
        test_acc = main(config=config)
        test_accs.append(test_acc)
    print('DEV({}) on CT dataset with drop rate ={} has mean test accuracy {} with std {}'.format('SO', drop_rate,
                                                                                                  np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
    logging.info('DEV({}) on CT dataset with drop rate ={} has mean test accuracy {} with std {}'.format('SO', drop_rate,
                                                                                                         np.mean(np.array(test_accs)), np.std(np.array(test_accs))))

# run baseline model
test_accs = []
with open('CharTrajectories/configs/train_dev.yaml') as file:
    config = ml_collections.ConfigDict(yaml.safe_load(file))
for model in baseline_models:

    for drop_rate in drop_rate_list:
        for i in range(runs):
            if model == 'LSTM':
                config = load_config(
                    'CharTrajectories/configs/train_lstm.yaml')

            elif model == 'signature':
                config = load_config('CharTrajectories/configs/train_sig.yaml')

            print(
                'Running {} model on drop rate ={}, run {}/5 '.format(model, drop_rate, i+1))
            config.drop_rate = drop_rate
            test_acc = main(config=config)
            test_accs.append(test_acc)
        print('{} model on CT dataset with drop rate ={} has mean test accuracy {} with std {}'.format(model, drop_rate,
                                                                                                       np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
        logging.info('{} model on CT dataset with drop rate ={} has mean test accuracy {} with std {}'.format(model, drop_rate,
                                                                                                              np.mean(np.array(test_accs)), np.std(np.array(test_accs))))

# run time-scale ditribution shift experiment
sample_rate = [1, 2]
for train_sr in sample_rate:
    for test_sr in sample_rate:
        config = load_config(
            'CharTrajectories/configs/train_dev.yaml')
        print('Running DEV({}) on train sample rate ={},test sample rate={} '.format(
            'SO', 1/train_sr, 1/test_sr))
        config.train_sr = train_sr
        config.test_sr = test_sr
        config.drop_rate = 0
        test_acc = main(config=config)
        test_accs.append(test_acc)
    print('DEV({}) on CT dataset with train sample rate ={},test sample rate = {} has mean test accuracy {} with std {}'.format('SO', 1/train_sr, 1/test_sr,
                                                                                                                                np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
    logging.info('DEV({}) on CT dataset with train sample rate ={},test sample rate = {} has mean test accuracy {} with std {}'.format('SO', 1/train_sr, 1/test_sr,
                                                                                                                                       np.mean(np.array(test_accs)), np.std(np.array(test_accs))))

for train_sr in sample_rate:
    for test_sr in sample_rate:
        config = load_config(
            'CharTrajectories/configs/train_lstm_dev.yaml')
        print('Running LSTM_DEV({}) on train sample rate ={},test sample rate={} '.format(
            'SO', 1/train_sr, 1/test_sr))
        config.train_sr = train_sr
        config.test_sr = test_sr
        config.drop_rate = 0
        test_acc = main(config=config)
        test_accs.append(test_acc)
    print('LSTM_DEV({}) on CT dataset with train sample rate ={},test sample rate = {} has mean test accuracy {} with std {}'.format('SO', 1/train_sr, 1/test_sr,
                                                                                                                                     np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
    logging.info('LSTM_DEV({}) on CT dataset with train sample rate ={},test sample rate = {} has mean test accuracy {} with std {}'.format('SO', 1/train_sr, 1/test_sr,
                                                                                                                                            np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
