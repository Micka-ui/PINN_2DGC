import tensorflow as tf
import keras


class PhysicsInformedNN(keras.Model):
    def __init__(self, Re, Sc, hidden_units, x_test, loss_fn, data_loss_w=100, training_metrics=[],
                 evaluation_metrics=[]):
        super(PhysicsInformedNN, self).__init__()
        ##Reynold number and Schmidt number
        self.data_loss_w = data_loss_w
        self.loss_fn = loss_fn
        self.Re = Re
        self.Sc = Sc
        self.ReSc = self.Re * self.Sc
        self.training_metrics = training_metrics
        self.evaluation_metrics = evaluation_metrics
        self.list_layers = [tf.keras.layers.Dense(hidden_units[0], activation='swish', input_shape=[x_test.shape[1]])] + \
                           [tf.keras.layers.Dense(hidden_unit,activation='swish') for hidden_unit in hidden_units] + \
                           [tf.keras.layers.Dense(4)]
        self.model_eqns = tf.keras.Sequential(self.list_layers)
        test = self.model_eqns(x_test)

    def __call__(self, inputs,training=True):
        pred = self.model_eqns(inputs, training=training)
        rho, u, v, p = [tf.reshape(pred[:, i], (-1, 1)) for i in range(4)]
        return rho, u, v, p

    def train_step(self, data):
        x_train, y_train = data

        # unpack x_train in x,y,t
        x, y, t = [tf.reshape(x_train[:, i], (-1, 1)) for i in range(3)]
        # data obs for rho and data obs for the equations
        rho_train, y_eqns = [tf.reshape(y_train[:, i], (-1, 1)) for i in range(2)]

        with tf.GradientTape() as tape:
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(y)
                t2.watch(t)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(x)
                    t1.watch(y)
                    t1.watch(t)
                    X = tf.stack([x[:, 0], y[:, 0], t[:, 0]], axis=1)
                    rho, u, v, p = self(X)

                ##rho 1st derivatives
                rho_x, rho_y, rho_t = [t1.gradient(rho, var) for var in [x, y, t]]
                ##u 1st derivatives
                u_x, u_y, u_t = [t1.gradient(u, var) for var in [x, y, t]]
                ##v 1st derivatives
                v_x, v_y, v_t = [t1.gradient(v, var) for var in [x, y, t]]
                ##p 1st derivatives
                p_x, p_y = [t1.gradient(p, var) for var in [x, y]]
            ##second derivatoves
            rho_xx, rho_yy, u_xx, u_yy, v_xx, v_yy = [t2.gradient(*ij) for ij in
                                                      zip([rho_x, rho_y, u_x, u_y, v_x, v_y], [x, y] * 3)]

            e1 = (u_t + u * u_x + v * u_y) + p_x - (1 / self.Re) * (u_xx + u_yy)
            e2 = (v_t + u * v_x + v * v_y) + p_y - (1 / self.Re) * (v_xx + v_yy) + rho
            e3 = u_x + v_y
            e4 = rho_t + u * rho_x + v * rho_y - (1 / self.ReSc) * (rho_xx + rho_yy)

            square_eqns = [tf.square(e_i) for e_i in [e1, e2, e3, e4]]

            square_data = tf.square(rho - rho_train)

            for e_i, metric in zip([square_data, e1, e2, e3, e4], self.training_metrics):
                metric.update_state(e_i, y_eqns)

            loss_eqns = self.loss_fn(tf.reduce_sum(square_eqns), y_eqns)
            loss_obs = self.loss_fn(square_data, y_eqns)
            loss = self.data_loss_w * loss_obs + loss_eqns

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        metrics = self.training_metrics
        return {m.name: m.result() for m in metrics}

    def test_step(self, data):
        x, y = data

        rho_valid, u_valid, v_valid = [tf.reshape(y[:, i], (-1, 1)) for i in range(3)]

        # Compute predictions
        rho_pred, u_pred, v_pred, p_pred = self(x, training=False)

        for metric, valid, pred in zip(self.evaluation_metrics, [rho_valid, u_valid, v_valid],
                                       [rho_pred, u_pred, v_pred]):
            metric.update_state(valid, pred)

        # Updates the metrics tracking the loss

        metrics = self.evaluation_metrics

        return {m.name: m.result() for m in metrics}
