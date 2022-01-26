from SpeechCommands.run import main
import numpy as np
from SpeechCommands.utils import load_config
import logging
baseline_models = ['LSTM', 'signature']

params = ['SO', 'SP']
runs = 5

test_accs = []
logging.basicConfig(filename='Speech_command_run_all',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
# run LSTM_DEV model
for i in range(runs):
    print('Running LSTM+DEV(SO), run {}/5 '.format(i+1))
    config = load_config('SpeechCommands/configs/train_lstm_dev.yaml')
    test_acc = main(config=config)
    test_accs.append(test_acc)
print('LSTM+DEV(SO) has mean test accuracy {} with std {}'.format(
    np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
logging.info('LSTM+DEV(SO) has mean test accuracy {} with std {}'.format(
    np.mean(np.array(test_accs)), np.std(np.array(test_accs))))

# run DEV model
test_accs = []
for i in range(runs):
    print('Running DEV({}), run {}/5 '.format('SO', i+1))
    config = load_config('SpeechCommands/configs/train_dev.yaml')
    test_acc = main(config=config)
    test_accs.append(test_acc)
print('DEV({}) has mean test accuracy {} with std {}'.format('SO',
                                                             np.mean(np.array(test_accs)), np.std(np.array(test_accs))))

logging.info('DEV({}) has mean test accuracy {} with std {}'.format('SO',
                                                                    np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
# run baseline model
test_accs = []

for model in baseline_models:

    for i in range(runs):
        if model == 'LSTM':
            config = load_config('SpeechCommands/configs/train_lstm.yaml')

        elif model == 'signature':
            config = load_config('SpeechCommands/configs/train_sig.yaml')

        print('Running {} model, run {}/5 '.format(model, i+1))
        test_acc = main(config=config)
        test_accs.append(test_acc)
    print('{} model has mean test accuracy {} with std {}'.format(model,
                                                                  np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
    logging.info('{} model has mean test accuracy {} with std {}'.format(model,
                                                                         np.mean(np.array(test_accs)), np.std(np.array(test_accs))))


# run linear model on signature and development
dev_size_list = [5, 10, 20, 30, 50, 100]

for dev_size in dev_size_list:
    config = load_config('SpeechCommands/configs/train_dev.yaml')
    config.n_hidden1 = dev_size
    if dev_size == 100:
        config.epochs = 30
        config.batch_size = 16
    elif dev_size == 50:
        config.batch_size = 64
    else:
        pass
    print('Running on Dev linear model with hidden size {}'.format(dev_size))
    test_acc = main(config=config)
    print('Dev linear model with hidden size {} has test accuracy = {}'.format(
        dev_size, test_acc))
    logging.info('Dev linear model with hidden size {} has test accuracy = {}'.format(
        dev_size, test_acc))

sig_depth = [1, 2, 3, 4]

for depth in sig_depth:
    config = load_config('SpeechCommands/configs/train_sig.yaml')
    config.depth = depth

    print('Running on signature linear model with depth {}'.format(depth))
    test_acc = main(config=config)
    print('signature linear model with depth {} has test accuracy = {}'.format(
        depth, test_acc))
    logging.info('signature linear model with depth {} has test accuracy = {}'.format(
        depth, test_acc))
