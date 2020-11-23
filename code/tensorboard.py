import tensorflow as tf
from tensorboard import main as tb
tf.flags.FLAGS.logdir = "/path/to/graphs/"
tb.main()