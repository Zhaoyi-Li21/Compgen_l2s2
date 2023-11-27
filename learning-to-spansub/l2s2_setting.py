from absl import app, flags, logging
FLAGS = flags.FLAGS
flags.DEFINE_integer('aug_encoder_n_embed',512,'')
flags.DEFINE_integer('aug_encoder_n_hidden',512,'')
flags.DEFINE_integer('aug_encoder_n_layer',2,'')
flags.DEFINE_float('aug_encoder_dropout',0.5,'')
flags.DEFINE_integer('aug_g_n_embed',512,'')
flags.DEFINE_integer('aug_f_n_embed',512,'')
