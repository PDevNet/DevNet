from Nbody.run import main
import numpy as np
from Nbody.utils import load_config
import logging
models = ['LSTM', 'LSTM_DEV']
params = ['SE', 'SO']
runs = 5

logging.basicConfig(filename='Nbody_run_all',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

test_accs = []
# run LSTM_DEV model
for param in params:
    for i in range(runs):
        config = load_config('Nbody/train.yaml')
        print('Running LSTM+DEV({}), run {}/5 '.format(param, i+1))
        config.model = 'LSTM_DEV'
        config.param = param
        test_acc = main(config=config)
        test_accs.append(test_acc)
    print('LSTM+DEV({}) has mean test accuracy {} with std {}'.format(param,
          np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
    logging.info('LSTM+DEV({}) has mean test accuracy {} with std {}'.format(param,
                                                                             np.mean(np.array(test_accs)), np.std(np.array(test_accs))))

test_accs = []
# run LSTM mode
for i in range(runs):
    config = load_config('Nbody/train.yaml')
    print('Running LSTM, run {}/5 '.format(i+1))
    config.model = 'LSTM'
    test_acc = main(config=config)
    test_accs.append(test_acc)
print('LSTM has mean test accuracy {} with std {}'.format(
    np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
logging.info('LSTM has mean test accuracy {} with std {}'.format(
    np.mean(np.array(test_accs)), np.std(np.array(test_accs))))
