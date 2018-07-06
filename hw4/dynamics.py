import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

def normalize(data, normalization):
    return (data - normalization[0]) / (normalization[1] + 1e-10)

def denormalize(data, normalization):
    return data * (normalization[1] + 1e-10) + normalization[0]

def batch_index(batch_size, chunk_size):
    ind =  [(acc, acc + batch_size) for acc in range(0, chunk_size, batch_size)]
    if ind[-1][1] < chunk_size:
      ind.append((ind[-1][1], chunk_size))
    return ind

class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        self.input_state = tf.placeholder(shape=(None, ob_dim), dtype=tf.float32)
        self.input_act = tf.placeholder(shape=(None, ac_dim), dtype=tf.float32)
        self.target_delta = tf.placeholder(shape=(None, ob_dim), dtype=tf.float32)

        self.dyn = build_mlp(tf.concat([self.input_state, self.input_act], axis=1),
                output_size=ob_dim,
                scope="NNDynamicsModel",
                n_layers=n_layers,
                size=size,
                activation=activation,
                output_activation=output_activation)

        self.normalization = normalization

        self.loss = tf.losses.mean_squared_error(labels=self.target_delta, predictions=self.dyn)
        self.update = tf.train.AdamOptimizer(learning_rate).minimize(loss=self.loss)
        self.iterations = iterations
        self.batch_size = batch_size
        self.sess = sess

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        obs = normalize(data["observations"], self.normalization["observations"])
        acts = normalize(data["actions"], self.normalization["actions"])
        deltas = normalize(data["next_observations"] - data["observations"], self.normalization["deltas"])
        chunk_size = len(data["observations"])

        batch_indexes = batch_index(self.batch_size, chunk_size)
        loss = None
        for epoch in range(self.iterations):
          if epoch % 20 == 0: print("Epoch {}/{}: Loss {}".format(epoch, self.iterations, loss))
          for _, (a,b) in enumerate(batch_indexes):
            _, loss = self.sess.run(
                [self.update, self.loss],
                feed_dict={
                    self.input_state: obs[a:b],
                    self.input_act: acts[a:b],
                    self.target_delta: deltas[a:b]
                    })

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        n_states = normalize(states, self.normalization["observations"])
        n_acts = normalize(actions, self.normalization["actions"])
        deltas = self.sess.run(self.dyn, feed_dict={self.input_state: n_states,
                                                    self.input_act: n_acts})
        deltas = denormalize(deltas, self.normalization["deltas"])
        return deltas + states
