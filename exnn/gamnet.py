import tensorflow as tf
from .base import BaseNet


class GAMNet(BaseNet):
 

    def __init__(self, subnet_arch=[10, 6], task_type="Regression",
                 activation_func=tf.tanh, batch_size=1000, training_epochs=10000, lr_bp=0.001,
                 beta_threshold=0.05, tuning_epochs=500, l1_subnet=0.001, l2_smooth=0.000001,
                 verbose=False, val_ratio=0.2, early_stop_thres=1000):

        super(GAMNet, self).__init__(
                             subnet_num=1,
                             subnet_arch=subnet_arch,
                             task_type=task_type,
                             proj_method="gam",
                             activation_func=activation_func,
                             bn_flag=True,
                             lr_bp=lr_bp,
                             l1_proj=0,
                             l1_subnet=l1_subnet,
                             l2_smooth=l2_smooth,
                             batch_size=batch_size,
                             training_epochs=training_epochs,
                             tuning_epochs=tuning_epochs,
                             beta_threshold=beta_threshold,
                             verbose=verbose,
                             val_ratio=val_ratio,
                             early_stop_thres=early_stop_thres)

    @tf.function
    def train_step_init(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, training=True)
            pred_loss = self.loss_fn(labels, pred)
            regularization_loss = tf.math.add_n(self.output_layer.losses)
            total_loss = pred_loss + regularization_loss
            if self.l2_smooth > 0:
                smoothness_loss = self.subnet_blocks.smooth_loss
                total_loss += smoothness_loss

        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(self.trainable_weights)):
            if self.trainable_weights[i].name != self.proj_layer.weights[0].name:
                train_weights_list.append(self.trainable_weights[i])

        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    @tf.function
    def train_step_finetune(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, training=True)
            pred_loss = self.loss_fn(labels, pred)
            total_loss = pred_loss
            if self.l2_smooth > 0:
                smoothness_loss = self.subnet_blocks.smooth_loss
                total_loss += smoothness_loss

        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(self.trainable_weights)):
            if self.trainable_weights[i].name != self.proj_layer.weights[0].name:
                train_weights_list.append(self.trainable_weights[i])

        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))
