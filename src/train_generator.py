import math

import torch
from torch import nn

import lib

from src.model import Generator


def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))

    generator = Generator(input_length)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    #loss
    mse_loss = torch.nn.MSELoss()

    # training data
    training_data = lib.data.get_training_set()

    for i in range(training_steps):

        for noisy_image, real_image in training_data:

            # zero the gradients on each iteration
            generator_optimizer.zero_grad()

            generated_data = generator(noisy_image)

            true_data = torch.tensor(real_image).float()

            generator_loss = mse_loss(generated_data, true_data)
            generator_loss.backward(generator_loss)
            generator_optimizer.step()

            generator.plotError()



