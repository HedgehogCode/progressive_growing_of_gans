import pickle
import os
import numpy as np
import tensorflow as tf
import PIL.Image

model = 'models/karras2018iclr-lsun-cat-256x256.pkl'
seed = 9
number = 10
output_dir = 'results'
output_prefix = 'cats'

# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open(model, 'rb') as file:
    G, D, Gs = pickle.load(file)

# Generate latent vectors.
latents = np.random.RandomState(seed).randn(number, *Gs.input_shapes[0][1:]) # random latents

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save(os.path.join(output_dir, output_prefix + '_%d.png' % idx))