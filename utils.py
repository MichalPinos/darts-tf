import functools
import tensorflow as tf

def activation_fn(features, act_fn):
  if act_fn in ('silu', 'swish'):
    return tf.nn.swish(features)
  elif act_fn == 'silu_native':
    return features * tf.sigmoid(features)
  elif act_fn == 'hswish':
    return features * tf.nn.relu6(features + 3) / 6
  elif act_fn == 'relu':
    return tf.nn.relu(features)
  elif act_fn == 'relu6':
    return tf.nn.relu6(features)
  elif act_fn == 'elu':
    return tf.nn.elu(features)
  elif act_fn == 'leaky_relu':
    return tf.nn.leaky_relu(features)
  elif act_fn == 'selu':
    return tf.nn.selu(features)
  elif act_fn == 'mish':
    return features * tf.math.tanh(tf.math.softplus(features))
  else:
    raise ValueError('Unsupported act_fn {}'.format(act_fn))


def get_act_fn(act_fn):
  if not act_fn:
    return tf.nn.relu
  if isinstance(act_fn, str):
    return functools.partial(activation_fn, act_fn=act_fn)
  return act_fn
