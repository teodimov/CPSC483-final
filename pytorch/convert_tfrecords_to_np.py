import os
import json
import numpy as np
import tensorflow.compat.v1 as tf
import reading_utils

tf.disable_v2_behavior()

def convert_split(data_path, split, metadata):
    """Convert a TFRecord split into a list of trajectories as numpy arrays."""
    file_path = os.path.join(data_path, f'{split}.tfrecord')
    dataset = tf.data.TFRecordDataset([file_path])

    def _parse_fn(example_proto):
        context, parsed_features = reading_utils.parse_serialized_simulation_example(example_proto, metadata)
        return context, parsed_features

    dataset = dataset.map(_parse_fn)

    # Note: Loads EVERYTHING into memory --> TO DO: Stream and save each example
    trajectories = []
    with tf.Session() as sess:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        try:
            while True:
                c, f = sess.run(next_element)
                # c['particle_type']: [num_particles]
                # f['position']: [sequence_length+1, num_particles, dim]
                # optional f['step_context']: [sequence_length+1, context_feat_len]

                trajectory = {
                    'particle_type': c['particle_type'],
                    'position': f['position']
                }
                if 'step_context' in f:
                    trajectory['step_context'] = f['step_context']
                trajectories.append(trajectory)
        except tf.errors.OutOfRangeError:
            pass

    return trajectories

def main():
    data_path = "/tmp/datasets/WaterDropSample"  #? MUST MODIFY FOR DATASET
    with open(os.path.join(data_path, 'metadata.json'), 'r') as fp:
        metadata = json.load(fp)

    for split in ['train', 'valid', 'test']:
        trajectories = convert_split(data_path, split, metadata)
        out_path = os.path.join(data_path, f"{split}_data.npz")
        np.savez_compressed(out_path, trajectories=trajectories)
        print(f"Saved {split} split with {len(trajectories)} trajectories.")

if __name__ == "__main__":
    main()
