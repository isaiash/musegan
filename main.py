import importlib
import numpy as np
import tensorflow as tf
from config import CONFIG

MODELS = importlib.import_module(
    '.'.join(('musegan', CONFIG['exp']['model'], 'models')))

def load_data():
    """Load and return the training data."""
    print('[*] Loading data...')

    # Load data from SharedArray
    if CONFIG['data']['training_data_location'] == 'sa':
        import SharedArray as sa
        x_train = sa.attach(CONFIG['data']['training_data'])

    # Load data from hard disk
    elif CONFIG['data']['training_data_location'] == 'hd':
        if os.path.isabs(CONFIG['data']['training_data_location']):
            x_train = np.load(CONFIG['data']['training_data'])
        else:
            filepath = os.path.abspath(os.path.join(
                os.path.realpath(__file__), 'training_data',
                CONFIG['data']['training_data']))
            x_train = np.load(filepath)

    # Reshape data
    x_train = x_train.reshape(
        -1, CONFIG['model']['num_bar'], CONFIG['model']['num_timestep'],
        CONFIG['model']['num_pitch'], CONFIG['model']['num_track'])
    print('Training set size:', len(x_train))

    return x_train

def main():
    """Main function."""
    if CONFIG['exp']['model'] not in ('musegan', 'bmusegan'):
        raise ValueError("Unrecognizable model name")

    print("Start experiment: {}".format(CONFIG['exp']['exp_name']))

    # Load training data
    x_train = load_data()

    # Open TensorFlow session
    with tf.Session(config=CONFIG['tensorflow']) as sess:
        if CONFIG['exp']['model'] == 'musegan':

            # Create model
            gan = MODELS.GAN(sess, CONFIG['model'])

            # Initialize all variables
            gan.init_all()

            # Load pretrained model if given
            if CONFIG['exp']['pretrained_dir'] is not None:
                gan.load_latest(CONFIG['exp']['pretrained_dir'])

            # Train the model
            gan.train(x_train, CONFIG['train'])

if __name__ == '__main__':
    main()
