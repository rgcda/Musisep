# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modified by Soeren Schulze in 2021

"""Adamax for dictionary learning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.Adamax')
class AdamaxDict(optimizer_v2.OptimizerV2):
  """
  Optimizer that implements the Adamax algorithm but uses the maximum over
  the specified axis in the denominator.

  Parameters
  ----------
  learning_rate : float
      Step size for training
  beta_1 : float
      Inertia for the numerator
  beta_2 : float
      Inertia for the denominator
  epsilon : float
      Additive constant in the denominator to avoid division by 0
  axis : int or sequence of int
      Axis in the variable to take the maximum over
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               axis=[],
               name='Adamax',
               **kwargs):
    super(AdamaxDict, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.axis = axis

  def _create_slots(self, var_list):
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')  # Create slots for the first moments.
    for var in var_list:
      self.add_slot(var, 'v')  # Create slots for the second moments.

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

    grad_max = math_ops.reduce_max(math_ops.abs(grad),
                                   axis=self.axis, keepdims=True)

    m.assign_add((grad - m) * (1 - beta_1_t))
    v.assign(math_ops.maximum(v * beta_2_t, grad_max))
    var.assign_sub(lr_t / (1 - beta_1_power) * (m / (v + epsilon_t)))

  def get_config(self):
    config = super(AdamaxDict, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
    })
    return config
