from datetime import datetime

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('DATA_PATH', "dataset/DATASET", "")
flags.DEFINE_string('LABEL_PATH', "dataset/DATASET", "")
flags.DEFINE_string('DICT_PATH', "dictionary/DATASET", "")

flags.DEFINE_integer('VOCAB_SIZE', 20000, '')
flags.DEFINE_integer('BATCH_SIZE', 32, '')
flags.DEFINE_integer('SEQ_LEN', 60, '')
flags.DEFINE_integer('EPOCH', 50, '')
flags.DEFINE_integer('BATCHES_PER_EPOCH', 1000, '')

flags.DEFINE_string('VAE_NAME', 'Simple_VAE', '')
flags.DEFINE_string('ENCODER_NAME', 'Encoder_vae', '')
flags.DEFINE_string('DECODER_NAME', 'Decoder_vae', '')

flags.DEFINE_integer('DROPWORD_KEEP', 0.62, '')
flags.DEFINE_integer('ENCODER_DROPOUT_KEEP', 1.0, '')
flags.DEFINE_integer('DECODER_DROPOUT_KEEP', 1.0, '')
flags.DEFINE_integer('LEARNING_RATE', 0.001, '')
flags.DEFINE_integer('LR_DECAY_START', 10, '')
flags.DEFINE_integer('MAX_GRAD', 5.0, '')

flags.DEFINE_integer('EMBED_SIZE', 353, '')
flags.DEFINE_integer('LATENT_VARIABLE_SIZE', 13, '')

flags.DEFINE_integer('RNN_NUM', 1, '')
flags.DEFINE_integer('RNN_SIZE', 191, '')

flags.DEFINE_string('LOG_DIR', "log/log" + datetime.now().strftime("%y%m%d-%H%M"), "")

FLAGS = flags.FLAGS
