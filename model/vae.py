import sys

import tensorflow as tf
import numpy as np

sys.path.append("../")

from config import FLAGS
from encoder import Encoder
from decoder import Decoder


class Simple_VAE(object):
    def __init__(self, batchloader, is_training=True, ru=False):
        self.batchloader = batchloader
        self.ru = ru
        self.is_training = is_training

        self.lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        with tf.name_scope("Placeholders"):
            self.encoder_input = tf.placeholder(tf.int64,
                                                shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                                name="encoder_input")

            self.decoder_input = tf.placeholder(tf.int64,
                                                shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                                name="decoder_input")

            self.target = tf.placeholder(tf.int64,
                                         shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                         name="target")

            encoder_input_t = tf.transpose(self.encoder_input, perm=[1, 0])
            self.encoder_input_list = []
            decoder_input_t = tf.transpose(self.decoder_input, perm=[1, 0])
            self.decoder_input_list = []
            target_t = tf.transpose(self.target, perm=[1, 0])
            self.target_list = []

            self.step = tf.placeholder(tf.float32, shape=(), name="step")

            for i in range(FLAGS.SEQ_LEN):
                self.encoder_input_list.append(encoder_input_t[i])
                assert self.encoder_input_list[i].shape == (FLAGS.BATCH_SIZE)

                self.decoder_input_list.append(decoder_input_t[i])
                assert self.decoder_input_list[i].shape == (FLAGS.BATCH_SIZE)

                self.target_list.append(target_t[i])
                assert self.target_list[i].shape == (FLAGS.BATCH_SIZE)


        with tf.variable_scope("Embedding"):
            self.embedding = tf.get_variable(name="embedding",
                                             shape=[FLAGS.VOCAB_SIZE, FLAGS.EMBED_SIZE],
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

        with tf.variable_scope("Encoder"):
            self.encoder = Encoder[FLAGS.ENCODER_NAME](
                                   self.embedding,
                                   self.encoder_input_list,
                                   is_training = self.is_training,
                                   ru = self.ru)

        with tf.name_scope("Latent_variables"):
            if self.is_training:
                self.latent_variables = self.encoder.latent_variables
            else:
                self.latent_variables = tf.placeholder(tf.float32,
                                                       shape=(FLAGS.BATCH_SIZE,
                                                              FLAGS.LATENT_VARIABLE_SIZE),
                                                       name="latent_variables_input")

        with tf.variable_scope("Decoder"):
            self.decoder = Decoder[FLAGS.DECODER_NAME](
                                           self.decoder_input_list,
                                           self.latent_variables,
                                           self.embedding,
                                           self.batchloader,
                                           is_training = self.is_training,
                                           ru = self.ru)

        with tf.name_scope("Loss"):
            self.logits = self.decoder.logits

            self.kld = tf.reduce_mean(-0.5 *
                                      tf.reduce_sum(self.encoder.logvar
                                                    - tf.square(self.encoder.mu)
                                                    - tf.exp(self.encoder.logvar)
                                                    + 1,
                                                    axis=1))
            self.kld_weight = tf.placeholder(tf.float32, shape=(), name="kld_weight")

            reconst_losses = [tf.nn.sparse_softmax_cross_entropy_with_logits( \
                                                    logits=logits, labels=targets) \
                                        for logits, targets in zip(self.logits, self.target_list)]
            self.reconst_loss = tf.reduce_mean(reconst_losses) * FLAGS.SEQ_LEN

            self.loss = self.reconst_loss + self.kld_weight * self.kld

        with tf.name_scope("Summary"):
            if is_training:
                loss_summary = tf.summary.scalar("loss", self.loss, family="train_loss")
                reconst_loss_summary = tf.summary.scalar("reconst_loss", self.reconst_loss, family="train_loss")
                kld_summary = tf.summary.scalar("kld", self.kld, family="kld")
                kld_weight_summary = tf.summary.scalar("kld_weight", self.kld_weight, family="parameters")
                mu_summary = tf.summary.histogram("mu", tf.reduce_mean(self.encoder.mu, 0))
                var_summary = tf.summary.histogram("var", tf.reduce_mean(tf.exp(self.encoder.logvar), 0))
                lr_summary = tf.summary.scalar("lr", self.lr, family="parameters")

                self.merged_summary = tf.summary.merge([loss_summary, reconst_loss_summary, kld_summary,
                                                        kld_weight_summary, mu_summary, var_summary,
                                                        lr_summary])
            else:
                valid_reconst_loss_summary = tf.summary.scalar("valid_reconst_loss", self.reconst_loss, family="valid_loss")

                self.merged_summary = tf.summary.merge([valid_reconst_loss_summary])

        if(self.is_training):
            tvars = tf.trainable_variables()
            with tf.name_scope("Optimizer"):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                  FLAGS.MAX_GRAD)
                optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

                self.train_op = optimizer.apply_gradients(zip(grads, tvars))


VAE = {
    "Simple_VAE": Simple_VAE,
}
