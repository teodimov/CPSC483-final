import numpy as np
import functools
import tensorflow as tf
import os
import json
import argparse
import pickle

_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}

def convert_to_tensor(values, encoded_dtype):
    """
    Convert encoded string values to tensor of specified dtype
    
    Args:
        values (tf.Tensor): Tensor of encoded string values
        encoded_dtype (type): Numpy dtype to decode to
    
    Returns:
        tf.Tensor: Decoded tensor of specified dtype
    """
    # Decode string values and convert to specified dtype
    decoded_values = tf.io.decode_raw(values, out_type=encoded_dtype)
    return decoded_values

def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def parse_serialized_simulation_example(example_proto, metadata):
    if 'context_mean' in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION
    
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description)
    
    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

    position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]
    parsed_features['position'] = tf.reshape(parsed_features['position'],
                                             position_shape)

    sequence_length = metadata['sequence_length'] + 1
    if 'context_mean' in metadata:
        context_feat_len = len(metadata['context_mean'])
        parsed_features['step_context'] = tf.reshape(
            parsed_features['step_context'],
            [sequence_length, context_feat_len])

    convert_particle_type_fn = functools.partial(convert_to_tensor, encoded_dtype=np.int64)
    context['particle_type'] = tf.py_function(
        convert_particle_type_fn,
        inp=[context['particle_type'].values],
        Tout=[tf.int64])
    context['particle_type'] = tf.reshape(context['particle_type'], [-1])

    return context, parsed_features

def parse_arguments():
    parser = argparse.ArgumentParser("Transform options.")
    parser.add_argument('--dataset', default='Water', type=str, help='dataset to transform.')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'valid'], help='split to transform.')

    args = parser.parse_args()

    return args

def transform(args):
    metadata = _read_metadata(f"datasets/{args.dataset}")
    ds = tf.data.TFRecordDataset([f"datasets/{args.dataset}/{args.split}.tfrecord"])
    ds = ds.map(functools.partial(
        parse_serialized_simulation_example, metadata=metadata
    ))

    positions = []
    step_context = []
    particle_types = []
    
    for context, parsed_features in ds:
        particle_types.append(context['particle_type'].numpy())
        positions.append(parsed_features['position'].numpy())
        if 'step_context' in parsed_features:
            step_context.append(parsed_features['step_context'].numpy())
    
    os.makedirs(f'datasets/{args.dataset}/{args.split}', exist_ok=True)
    
    with open(f'datasets/{args.dataset}/{args.split}/positions.pkl', 'wb') as position_file:
        pickle.dump(positions, position_file)

    with open(f'datasets/{args.dataset}/{args.split}/particle_types.pkl', 'wb') as particle_file:
        pickle.dump(particle_types, particle_file)

    if len(step_context) != 0:
        with open(f'datasets/{args.dataset}/{args.split}/step_context.pkl', 'wb') as context_file:
            pickle.dump(step_context, context_file)

if __name__ == '__main__':
    transform(parse_arguments())
