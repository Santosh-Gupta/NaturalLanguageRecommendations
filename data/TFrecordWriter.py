import os
import tensorflow as tf
from tqdm import tqdm


class TFrecordWriter:
    def __init__(self,
                 n_samples,
                 n_shards,
                 output_dir='',
                 prefix=''):
        self.n_samples = n_samples
        self.n_shards = n_shards
        self.step_size = self.n_samples // self.n_shards + 1
        self.prefix = prefix
        self.output_dir = output_dir
        self.buffer = []
        self.file_count = 1

    def make_example(self, title, vector):
        feature = {
            'title': tf.train.Feature(int64_list=tf.train.Int64List(value=title)),
            'citation': tf.train.Feature(float_list=tf.train.FloatList(value=vector))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecord(self, tfrecord_path):
        print('writing {} samples in {}'.format(
            len(self.buffer), tfrecord_path))
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for (title, vector) in tqdm(self.buffer):
                example = self.make_example(title, vector)
                writer.write(example.SerializeToString())

    def push(self, title, vector):
        self.buffer.append([title, vector])
        if len(self.buffer) == self.step_size:
            fname = self.prefix + '_000' + str(self.file_count) + '.tfrecord'
            tfrecord_path = os.path.join(self.output_dir, fname)
            self.write_tfrecord(tfrecord_path)
            self.clear_buffer()
            self.file_count += 1

    def flush_last(self):
        if len(self.buffer):
            fname = self.prefix + '_000' + str(self.file_count) + '.tfrecord'
            tfrecord_path = os.path.join(self.output_dir, fname)
            self.write_tfrecord(tfrecord_path)

    def clear_buffer(self):
        self.buffer = []
