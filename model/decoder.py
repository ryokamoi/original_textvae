import sys

import tensorflow as tf

sys.path.append("../")

from config import FLAGS


class Decoder_vae(object):
    def __init__(self, decoder_input_list, latent_variables, embedding,
                 batchloader, is_training=True, ru=False):
        with tf.name_scope("decoder_input"):
            self.decoder_input_list = decoder_input_list
            self.latent_variables = latent_variables
            self.embedding = embedding
            self.batchloader = batchloader
            self.go_input = tf.constant(self.batchloader.go_input,
                                        dtype=tf.int32)

            self.is_training = is_training

        # rnn
        with tf.variable_scope("decoder_rnn"):
            with tf.variable_scope("rnn_input_weight"):
                self.rnn_input_W = tf.get_variable(name="rnn_input_W",
                                                   shape=(FLAGS.EMBED_SIZE + FLAGS.LATENT_VARIABLE_SIZE,
                                                          FLAGS.RNN_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                self.rnn_input_b = tf.get_variable(name="rnn_input_b",
                                                   shape=(FLAGS.RNN_SIZE),
                                                   dtype=tf.float32)

            with tf.variable_scope("decoder_rnn"):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(FLAGS.RNN_SIZE)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     output_keep_prob=FLAGS.DECODER_DROPOUT_KEEP)
                self.cell = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.RNN_NUM)

                self.init_states = [cell.zero_state(FLAGS.BATCH_SIZE, tf.float32)
                                    for _ in range(FLAGS.RNN_NUM)]
                self.states = [tf.placeholder(tf.float32,
                                              (FLAGS.BATCH_SIZE),
                                              name="state")
                               for _ in range(FLAGS.RNN_NUM)]

            with tf.variable_scope("decoder_rnn2vocab"):
                self.rnn2vocab_W = tf.get_variable(name="rnn2vocab_W",
                                                   shape=(FLAGS.RNN_SIZE, FLAGS.VOCAB_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                self.rnn2vocab_b = tf.get_variable(name="rnn2vocab_b",
                                                   shape=(FLAGS.VOCAB_SIZE),
                                                   dtype=tf.float32)

            with tf.name_scope("decoder_rnn_output"):
                if self.is_training:
                    self.logits = self.rnn_train_predict()
                else:
                    self.logits = self.rnn_valid_predict()


    # input text from dataset
    def rnn_train_predict(self):
        pred = []
        state = self.init_states
        for i in range(FLAGS.SEQ_LEN):
            with tf.name_scope("decoder_input_and_lv"):
                decoder_input = self.decoder_input_list[i]
                decoder_input_embedding = tf.nn.embedding_lookup(self.embedding, decoder_input)
                rnn_input = tf.concat([self.latent_variables, decoder_input_embedding], axis=1)
                assert rnn_input.shape == (FLAGS.BATCH_SIZE,
                                           FLAGS.LATENT_VARIABLE_SIZE + FLAGS.EMBED_SIZE)

            with tf.name_scope("rnn_input"):
                rnn_input = tf.matmul(rnn_input, self.rnn_input_W) + self.rnn_input_b
                assert rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            with tf.name_scope("rnn_predict"):
                step_pred, state = self.cell(rnn_input, state)
                assert step_pred.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

                step_word = tf.matmul(step_pred, self.rnn2vocab_W) + self.rnn2vocab_b
                assert step_word.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                pred.append(step_word)

        return pred

    # input text from previous output
    def rnn_valid_predict(self):
        pred = []
        state = self.init_states
        next_input = tf.nn.embedding_lookup(self.embedding, self.go_input)
        for i in range(FLAGS.SEQ_LEN):
            with tf.name_scope("decoder_input_embedding"):
                rnn_input = tf.concat([self.latent_variables, next_input], axis=1)
                assert rnn_input.shape == (FLAGS.BATCH_SIZE,
                                           FLAGS.LATENT_VARIABLE_SIZE + FLAGS.EMBED_SIZE)

            with tf.name_scope("rnn_input"):
                rnn_input = tf.matmul(rnn_input, self.rnn_input_W) + self.rnn_input_b
                assert rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            with tf.name_scope("rnn_input"):
                step_pred, state = self.cell(rnn_input, state)
                assert step_pred.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            with tf.name_scope("rnn_predict"):
                step_word = tf.matmul(step_pred, self.rnn2vocab_W) + self.rnn2vocab_b
                assert step_word.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                pred.append(step_word)

                next_symbol = tf.stop_gradient(tf.argmax(step_word, 1))
                next_input = tf.nn.embedding_lookup(self.embedding, next_symbol)

        return pred


Decoder= {
    "Decoder_vae": Decoder_vae,
}
