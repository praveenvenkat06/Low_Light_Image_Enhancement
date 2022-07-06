import math
import random

import torch
from torch import nn

import lib
from discriminator import Discriminator

def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))

    discriminator = Discriminator(input_length)
    discriminator_optimizer = torch.optim.Adam(Discriminator.parameters(), lr=0.001)

    #loss
    mse_loss = torch.nn.MSELoss()

    # training data
    training_data = lib.data.get_training_set()

    for i in range(training_steps):

        for noisy_image, real_image in training_data:

            # pick the noisy image or the real image at random
            # and feed it to the discriminator
            # back propagate according to the real and predicted labels

            # zero the gradients on each iteration
            discriminator_optimizer.zero_grad()

            if random.random() > 0.5:
                generated_labels = torch.tensor(Discriminator(noisy_image)).float()
                true_labels = torch.tensor(real_image).float()

                loss = mse_loss(generated_labels, true_labels)

                discriminator.backward(loss)
                discriminator_optimizer.step()

                discriminator.plotError()

            else:
                generated_labels = torch.tensor(Discriminator(real_image)).float()
                true_labels = torch.tensor(noisy_image).float()

                loss = mse_loss(generated_labels, true_labels)

                discriminator.backward(loss)
                discriminator_optimizer.step()

                discriminator.plotError()





