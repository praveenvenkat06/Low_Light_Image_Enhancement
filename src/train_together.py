import math

import torch
import torch.nn as nn
import lib.data

from src.discriminator import Discriminator
from src.model import Generator

from train_generator import train as train_gen
from train_discriminator import train as train_disc


def train_together(max_int: int = 128, batch_size: int = 16, training_steps: int = 10000):
    input_length = int(math.log(max_int, 2))

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    # training data
    training_data = lib.data.get_training_set()

    for i in range(training_steps):

        for noisy_image, real_image in training_data:

            # zero the gradients on each iteration
            generator_optimizer.zero_grad()

            generated_data = generator(noisy_image)

            true_labels = torch.tensor(real_image).float()

            # Train the generator
            generator_discriminator_out = discriminator(generated_data)
            generator_loss = loss(generator_discriminator_out, true_labels)
            generator.backward(generator_loss)
            generator_optimizer.step()

            # Train the discriminator on the true/generated data
            discriminator_optimizer.zero_grad()
            true_discriminator_out = discriminator(true_labels)
            true_discriminator_loss = loss(true_discriminator_out, generator_discriminator_out )
            discriminator.backward(true_discriminator_loss)
            discriminator_optimizer.step()

            generator.plotError()
            discriminator.plotError()

def train_from_scratch(max_int: int = 128, batch_size: int = 16, training_steps: int = 10000):
    await train_gen()
    await train_disc()

    train_together(max_int, batch_size, training_steps)