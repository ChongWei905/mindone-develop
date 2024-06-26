
import mindspore as ms

TRANSFER_MAP = {
    'running_mean': 'moving_mean',
    'running_var': 'moving_variance',
    'bn.weight': 'bn.gamma',
    'bn.bias': 'bn.beta',
    'Conv2d_1a_3x3.': 'conv1a.',
    'Conv2d_2a_3x3.': 'conv2a.',
    'Conv2d_2b_3x3.': 'conv2b.',
    'Conv2d_3b_1x1.': 'conv3b.',
    'Conv2d_4a_3x3.': 'conv4a.',
    'Mixed_5b.branch1x1.': 'inception5b.branch0.',
    'Mixed_5b.branch3x3dbl_1.': 'inception5b.branch2.0.',
    'Mixed_5b.branch3x3dbl_2.': 'inception5b.branch2.1.',
    'Mixed_5b.branch3x3dbl_3.': 'inception5b.branch2.2.',
    'Mixed_5b.branch5x5_1.': 'inception5b.branch1.0.',
    'Mixed_5b.branch5x5_2.': 'inception5b.branch1.1.',
    'Mixed_5b.branch_pool.': 'inception5b.branch_pool.1.',
    'Mixed_5c.branch1x1.': 'inception5c.branch0.',
    'Mixed_5c.branch3x3dbl_1.': 'inception5c.branch2.0.',
    'Mixed_5c.branch3x3dbl_2.': 'inception5c.branch2.1.',
    'Mixed_5c.branch3x3dbl_3.': 'inception5c.branch2.2.',
    'Mixed_5c.branch5x5_1.': 'inception5c.branch1.0.',
    'Mixed_5c.branch5x5_2.': 'inception5c.branch1.1.',
    'Mixed_5c.branch_pool.': 'inception5c.branch_pool.1.',
    'Mixed_5d.branch1x1.': 'inception5d.branch0.',
    'Mixed_5d.branch3x3dbl_1.': 'inception5d.branch2.0.',
    'Mixed_5d.branch3x3dbl_2.': 'inception5d.branch2.1.',
    'Mixed_5d.branch3x3dbl_3.': 'inception5d.branch2.2.',
    'Mixed_5d.branch5x5_1.': 'inception5d.branch1.0.',
    'Mixed_5d.branch5x5_2.': 'inception5d.branch1.1.',
    'Mixed_5d.branch_pool.': 'inception5d.branch_pool.1.',
    'Mixed_6a.branch3x3.': 'inception6a.branch0.',
    'Mixed_6a.branch3x3dbl_1.': 'inception6a.branch1.0.',
    'Mixed_6a.branch3x3dbl_2.': 'inception6a.branch1.1.',
    'Mixed_6a.branch3x3dbl_3.': 'inception6a.branch1.2.',
    'Mixed_6b.branch1x1.': 'inception6b.branch0.',
    'Mixed_6b.branch7x7_1.': 'inception6b.branch1.0.',
    'Mixed_6b.branch7x7_2.': 'inception6b.branch1.1.',
    'Mixed_6b.branch7x7_3.': 'inception6b.branch1.2.',
    'Mixed_6b.branch7x7dbl_1.': 'inception6b.branch2.0.',
    'Mixed_6b.branch7x7dbl_2.': 'inception6b.branch2.1.',
    'Mixed_6b.branch7x7dbl_3.': 'inception6b.branch2.2.',
    'Mixed_6b.branch7x7dbl_4.': 'inception6b.branch2.3.',
    'Mixed_6b.branch7x7dbl_5.': 'inception6b.branch2.4.',
    'Mixed_6b.branch_pool.': 'inception6b.branch_pool.1.',
    'Mixed_6c.branch1x1.': 'inception6c.branch0.',
    'Mixed_6c.branch7x7_1.': 'inception6c.branch1.0.',
    'Mixed_6c.branch7x7_2.': 'inception6c.branch1.1.',
    'Mixed_6c.branch7x7_3.': 'inception6c.branch1.2.',
    'Mixed_6c.branch7x7dbl_1.': 'inception6c.branch2.0.',
    'Mixed_6c.branch7x7dbl_2.': 'inception6c.branch2.1.',
    'Mixed_6c.branch7x7dbl_3.': 'inception6c.branch2.2.',
    'Mixed_6c.branch7x7dbl_4.': 'inception6c.branch2.3.',
    'Mixed_6c.branch7x7dbl_5.': 'inception6c.branch2.4.',
    'Mixed_6c.branch_pool.': 'inception6c.branch_pool.1.',
    'Mixed_6d.branch1x1.': 'inception6d.branch0.',
    'Mixed_6d.branch7x7_1.': 'inception6d.branch1.0.',
    'Mixed_6d.branch7x7_2.': 'inception6d.branch1.1.',
    'Mixed_6d.branch7x7_3.': 'inception6d.branch1.2.',
    'Mixed_6d.branch7x7dbl_1.': 'inception6d.branch2.0.',
    'Mixed_6d.branch7x7dbl_2.': 'inception6d.branch2.1.',
    'Mixed_6d.branch7x7dbl_3.': 'inception6d.branch2.2.',
    'Mixed_6d.branch7x7dbl_4.': 'inception6d.branch2.3.',
    'Mixed_6d.branch7x7dbl_5.': 'inception6d.branch2.4.',
    'Mixed_6d.branch_pool.': 'inception6d.branch_pool.1.',
    'Mixed_6e.branch1x1.': 'inception6e.branch0.',
    'Mixed_6e.branch7x7_1.': 'inception6e.branch1.0.',
    'Mixed_6e.branch7x7_2.': 'inception6e.branch1.1.',
    'Mixed_6e.branch7x7_3.': 'inception6e.branch1.2.',
    'Mixed_6e.branch7x7dbl_1.': 'inception6e.branch2.0.',
    'Mixed_6e.branch7x7dbl_2.': 'inception6e.branch2.1.',
    'Mixed_6e.branch7x7dbl_3.': 'inception6e.branch2.2.',
    'Mixed_6e.branch7x7dbl_4.': 'inception6e.branch2.3.',
    'Mixed_6e.branch7x7dbl_5.': 'inception6e.branch2.4.',
    'Mixed_6e.branch_pool.': 'inception6e.branch_pool.1.',
    'Mixed_7a.branch3x3_1.': 'inception7a.branch0.0.',
    'Mixed_7a.branch3x3_2.': 'inception7a.branch0.1.',
    'Mixed_7a.branch7x7x3_1.': 'inception7a.branch1.0.',
    'Mixed_7a.branch7x7x3_2.': 'inception7a.branch1.1.',
    'Mixed_7a.branch7x7x3_3.': 'inception7a.branch1.2.',
    'Mixed_7a.branch7x7x3_4.': 'inception7a.branch1.3.',
    'Mixed_7b.branch1x1.': 'inception7b.branch0.',
    'Mixed_7b.branch3x3_1.': 'inception7b.branch1.',
    'Mixed_7b.branch3x3_2a.': 'inception7b.branch1a.',
    'Mixed_7b.branch3x3_2b.': 'inception7b.branch1b.',
    'Mixed_7b.branch3x3dbl_1.': 'inception7b.branch2.0.',
    'Mixed_7b.branch3x3dbl_2.': 'inception7b.branch2.1.',
    'Mixed_7b.branch3x3dbl_3a.': 'inception7b.branch2a.',
    'Mixed_7b.branch3x3dbl_3b.': 'inception7b.branch2b.',
    'Mixed_7b.branch_pool.': 'inception7b.branch_pool.1.',
    'Mixed_7c.branch1x1.': 'inception7c.branch0.',
    'Mixed_7c.branch3x3_1.': 'inception7c.branch1.',
    'Mixed_7c.branch3x3_2a.': 'inception7c.branch1a.',
    'Mixed_7c.branch3x3_2b.': 'inception7c.branch1b.',
    'Mixed_7c.branch3x3dbl_1.': 'inception7c.branch2.0.',
    'Mixed_7c.branch3x3dbl_2.': 'inception7c.branch2.1.',
    'Mixed_7c.branch3x3dbl_3a.': 'inception7c.branch2a.',
    'Mixed_7c.branch3x3dbl_3b.': 'inception7c.branch2b.',
    'Mixed_7c.branch_pool.': 'inception7c.branch_pool.1.',
    'fc.bias': 'classifier.bias',
    'fc.weight': 'classifier.weight'
}


def transfer_torch_inception_weights(state_dict):
    for key in list(state_dict.keys()):
        value = state_dict.pop(key)
        new_key = key
        for name in list(TRANSFER_MAP.keys()):
            if name in new_key:
                new_key = new_key.replace(name, TRANSFER_MAP[name])
        state_dict[new_key] = ms.Parameter(ms.Tensor(value.numpy()))

    return state_dict
