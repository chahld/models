# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python convert_data_directory.py \
    --source_dir=/tmp/pictures \
    --dataset_dir=/tmp/picture_dataset \
    --num_validation=500 \
    --num_shards=5 \

```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_directory

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'source_dir',
    None,
    ('The source directory with subdirectory per label containing image files for that label'))

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_integer(
    'num_validation',
    None,
    'number of validation examples, defaults to 4%')

tf.app.flags.DEFINE_integer(
    'num_shards',
    5,
    'number of shard files, defaults to 5')

def main(_):
  if not FLAGS.source_dir:
    raise ValueError('You must supply the source directory with --source_dir')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  convert_directory.run(FLAGS.source_dir, FLAGS.dataset_dir, FLAGS.num_validation, FLAGS.num_shards)

if __name__ == '__main__':
  tf.app.run()
