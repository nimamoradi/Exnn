import tensorflow as tf
from .base import BaseNet
import numpy as np

class ExNN(BaseNet):
    """
    Enhanced explainable neural network (ExNN) based on sparse, orthogonal and smooth constraints.
    """

    def __init__(self, subnet_num, subnet_arch=[10, 6], task_type="Regression",
                 activation_func=tf.tanh, batch_size=1000, training_epochs=10000, lr_bp=0.001, lr_cl=0.1,
                 beta_threshold=0.05, tuning_epochs=500, l1_proj=0.001, l1_subnet=0.001, l2_smooth=0.000001,
                 verbose=False, val_ratio=0.2, early_stop_thres=1000, random_state=0):

        super(ExNN, self).__init__(
                             subnet_num=subnet_num,
                             subnet_arch=subnet_arch,
                             task_type=task_type,
                             proj_method="orthogonal",
                             activation_func=activation_func,
                             bn_flag=True,
                             lr_bp=lr_bp,
                             l1_proj=l1_proj,
                             l1_subnet=l1_subnet,
                             l2_smooth=l2_smooth,
                             batch_size=batch_size,
                             training_epochs=training_epochs,
                             tuning_epochs=tuning_epochs,
                             beta_threshold=beta_threshold,
                             verbose=verbose,
                             val_ratio=val_ratio,
                             early_stop_thres=early_stop_thres,
                             random_state=random_state)
        self.lr_cl = lr_cl

    @tf.function
    def train_step_init(self, inputs, labels):
        with tf.GradientTape() as tape_cl:
            with tf.GradientTape() as tape_bp:
                print(inputs)
                pred = self.__call__(np.int32(inputs), training=True)
                pred_loss = self.loss_fn(labels, pred)
                regularization_loss = tf.math.add_n(self.proj_layer.losses + self.output_layer.losses)
                cl_loss = pred_loss + regularization_loss
                bp_loss = pred_loss + regularization_loss
                if self.l2_smooth > 0:
                    smoothness_loss = self.subnet_blocks.smooth_loss
                    bp_loss += smoothness_loss

        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(self.trainable_weights)):
            if self.trainable_weights[i].name != self.proj_layer.weights[0].name:
                train_weights_list.append(self.trainable_weights[i])

        grad_proj = tape_cl.gradient(cl_loss, self.proj_layer.weights)
        grad_nets = tape_bp.gradient(bp_loss, train_weights_list)

        in_shape = self.proj_layer.weights[0].shape[0]
        matrix_a = (tf.matmul(grad_proj[0], tf.transpose(self.proj_layer.weights[0]))
                    - tf.matmul(self.proj_layer.weights[0], tf.transpose(grad_proj[0])))
        matrix_q = tf.matmul(tf.linalg.inv(tf.eye(in_shape) + tf.multiply(self.lr_cl / 2, matrix_a)),
                             (tf.eye(in_shape) - tf.multiply(self.lr_cl / 2, matrix_a)))
        self.proj_layer.weights[0].assign(tf.matmul(matrix_q, self.proj_layer.weights[0]))
        self.optimizer.apply_gradients(zip(grad_nets, train_weights_list))

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
