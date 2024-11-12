import numpy as np
import torch

from datasets.utd_mhad import UTDInertialInstance

class Jittering():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        noise = np.random.normal(loc=0, scale=self.sigma, size=x.shape)
        x = x + torch.tensor(noise).float()
        return x

class Scaling():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        factor = np.random.normal(loc=1., scale=self.sigma, size=(x.shape))
        x = x * factor
        return x

class Rotation():
    def __init__(self):
        pass

    def __call__(self, x):
        flip = torch.tensor(np.random.choice([-1, 1], size=(x.shape)))
        return flip * x

class ChannelShuffle():
    def __init__(self):
        pass

    def __call__(self, x):
        rotate_axis = np.arange(x.shape[1])
        np.random.shuffle(rotate_axis)
        return x[:, rotate_axis]

class Permutation():
    def __init__(self, max_segments=5):
        self.max_segments = max_segments

    def __call__(self, x):
        orig_steps = np.arange(x.shape[0])
        num_segs = np.random.randint(1, self.max_segments)
        
        if num_segs > 1:
            # Ensure all segments are equal length
            seg_length = len(orig_steps) // num_segs
            splits = [orig_steps[i:i + seg_length] for i in range(0, len(orig_steps), seg_length)]
            
            # Handle any remaining elements
            if len(splits[-1]) < seg_length:
                splits[-2] = np.concatenate([splits[-2], splits[-1]])
                splits.pop()
            
            # Permute and reconstruct
            perm_splits = np.random.permutation(splits)
            warp = np.concatenate(perm_splits)
            ret = x[warp]
        else:
            ret = x
            
        # Ensure output length matches input length
        if len(ret) != len(x):
            ret = ret[:len(x)]
            
        return ret

    def apply_augmentation(self, x):
        # Make sure output length matches input length
        output_length = len(x)
        # Apply augmentations while preserving length
        augmented = self(x)
        # Resample/pad/truncate to match original length
        augmented = resample_to_length(augmented, output_length)
        return augmented


if __name__ == '__main__':
    test_signal = UTDInertialInstance('/home/data/multimodal_har_datasets/utd_mhad/Inertial/a23_s5_t4_inertial.mat').signal
    print("Original: ", test_signal[0])

    jittered = Jittering(0.05)(test_signal)
    print("Jittered: ", jittered[0])

    scaled = Scaling(0.9)(test_signal)
    print("Scaled:   ", scaled[0])

    rotated = Rotation()(test_signal)
    print("Rotated:  ", rotated[0])

    permuted = Permutation()(test_signal)
    print("Permuted: ", permuted[0])

    shuffled = ChannelShuffle()(test_signal)
    print("Shuffled: ", shuffled[0])
