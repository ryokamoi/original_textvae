import numpy as np

import tensorflow as tf

from config import FLAGS

class RNN(object):
    def __init__(self, train=True, ru=False):
        with tf.variable_scope("RNN_Network", reuse=ru):
            with tf.name_scope("Placeholders"):
                self.decoder_input = tf.placeholder(tf.int32,
                                                    shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                                    name="decoder_input")
                self.decoder_input_t = tf.transpose(self.decoder_input, perm=[1, 0])

                self.target = tf.placeholder(tf.int32,
                                             shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                             name="target")
                self.target_t = tf.transpose(self.target, perm=[1, 0])

                self.mask = tf.placeholder(tf.float32,
                                           shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                           name="mask")
                self.mask_t = tf.transpose(self.mask, perm=[1, 0])

                self.decoder_input_list = []
                self.target_list = []
                self.mask_list = []
                for i in range(FLAGS.SEQ_LEN):
                    self.decoder_input_list.append(self.decoder_input_t[i])
                    assert self.decoder_input_list[i].shape == (FLAGS.BATCH_SIZE)

                    self.target_list.append(self.target_t[i])
                    assert self.target_list[i].shape == (FLAGS.BATCH_SIZE)

                    self.mask_list.append(self.mask_t[i])
                    assert self.mask_list[i].shape == (FLAGS.BATCH_SIZE)

            with tf.name_scope("Embedding"):
                self.embedding = tf.Variable(tf.random_uniform((FLAGS.VOCAB_SIZE, FLAGS.EMBED_SIZE)),
                                             dtype=tf.float32,
                                             name="embedding")

            with tf.name_scope("RNN"):
                cell = tf.nn.rnn_cell.GRUCell(FLAGS.RNN_HIDDEN_SIZE)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     output_keep_prob=FLAGS.DROPOUT_KEEP)
                self.cell = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.RNN_NUM)

                self.init_states = [cell.zero_state(FLAGS.BATCH_SIZE, tf.float32)
                                    for _ in range(FLAGS.RNN_NUM)]
                self.states = [tf.placeholder(tf.float32, (FLAGS.BATCH_SIZE),
                                              name="state")
                               for _ in range(FLAGS.RNN_NUM)]

            with tf.name_scope("Linear"):
                self.rnn2vocab_W = tf.Variable(tf.random_uniform((FLAGS.RNN_HIDDEN_SIZE,
                                                                  FLAGS.VOCAB_SIZE),
                                                                  dtype=tf.float32,
                                                                  name="rnn2vocab_W"))
                self.rnn2vocab_b = tf.Variable(tf.zeros(FLAGS.VOCAB_SIZE),
                                               dtype=tf.float32,
                                               name="rnn2vocab_b")

            with tf.name_scope("Loss"):
                self.train_logits = self.predict_with_gt()
                self.inference_logits = self.predict_without_gt()

                self.loss = tf.reduce_mean(tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                                            self.train_logits,
                                            self.target_list,
                                            [tf.ones(FLAGS.BATCH_SIZE) for _ in range(FLAGS.SEQ_LEN)]))

                tf.summary.scalar("loss", self.loss)

            with tf.name_scope("Optimizer"):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                  FLAGS.MAX_GRAD)
                optimizer = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            self.merged_summary = tf.summary.merge_all()

    def predict_with_gt(self):
        pred = []
        state = self.init_states
        for i in range(FLAGS.SEQ_LEN):
            input = self.decoder_input_t[i]
            assert input.shape == (FLAGS.BATCH_SIZE)

            rnn_input = tf.nn.embedding_lookup(self.embedding, input)
            assert rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.EMBED_SIZE)

            step_pred, state = self.cell(rnn_input, state)
            assert step_pred.shape == (FLAGS.BATCH_SIZE, FLAGS.EMBED_SIZE)

            step_word = tf.matmul(step_pred, self.rnn2vocab_W) + self.rnn2vocab_b
            assert step_word.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

            pred.append(step_word)

        return pred

    def predict_without_gt(self):
        pred = []
        state = self.init_states
        next_input = tf.nn.embedding_lookup(self.embedding, self.decoder_input_t[0])
        for i in range(FLAGS.SEQ_LEN):
            assert next_input.shape == (FLAGS.BATCH_SIZE, FLAGS.EMBED_SIZE)

            step_pred, state = self.cell(next_input, state)
            assert step_pred.shape == (FLAGS.BATCH_SIZE, FLAGS.EMBED_SIZE)

            step_word = tf.matmul(step_pred, self.rnn2vocab_W) + self.rnn2vocab_b
            assert step_word.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

            next_symbol = tf.stop_gradient(tf.argmax(step_word, 1))
            next_input = tf.nn.embedding_lookup(self.embedding, next_symbol)

            pred.append(step_word)

        return pred
