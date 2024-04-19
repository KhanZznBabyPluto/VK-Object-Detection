import tensorflow as tf

class RMSE(tf.keras.metrics.Metric):

    def __init__(self, name='rmse', **kwargs):
        super(RMSE, self).__init__(name=name, **kwargs)
        self.mse = self.add_weight(name='mse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.int32)
        squared_error = tf.square(y_true - y_pred)
        mse = tf.reduce_mean(squared_error)
        self.mse.assign_add(mse)
        self.count.assign_add(1)

    def result(self):
        return tf.sqrt(self.mse / self.count)

    def reset_states(self):
        self.mse.assign(0)
        self.count.assign(0)
