import tensorflow as tf

from collections.abc import Callable
from typing import Dict


class Critic(tf.keras.Model):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__(name="critic", *args, **kwargs)
        self.hidden_layer = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        hidden = self.hidden_layer(x)
        return self.output_layer(hidden)


class Generator(tf.keras.Model):
    def __init__(self, output_dim, hidden_dim, *args, **kwargs):
        super().__init__(name="generator", *args, **kwargs)
        self.hidden_layer = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.output_layer = tf.keras.layers.Dense(output_dim, activation="sigmoid")

    def call(self, x):
        hidden = self.hidden_layer(x)
        return self.output_layer(hidden)


class WGAN(tf.keras.Model):
    def __init__(
        self,
        critic: tf.keras.Model,
        generator: tf.keras.Model,
        latent_dim: int,
        n_critic: int = 5,
        lambda_: float = 10.0,
    ):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_ = lambda_

    def compile(
        self,
        critic_optimizer: tf.keras.optimizers.Adam,
        generator_optimizer: tf.keras.optimizers.Adam,
        critic_loss_fn: Callable[[tf.Tensor, tf.Tensor], float],
        generator_loss_fn: Callable[[tf.Tensor], float],
    ) -> None:
        super().compile()
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        self.critic_loss_fn = critic_loss_fn
        self.generator_loss_fn = generator_loss_fn

    def gradient_penalty(
        self, batch_size: int, real_samples: tf.Tensor, fake_samples: tf.Tensor
    ) -> tf.Tensor:
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the critic loss.
        """
        # Get the interpolated samples

        # from https://keras.io/examples/generative/wgan_gp/#create-the-wgangp-model
        # noise = tf.random.normal((batch_size, 1), 0.0, 1.0)
        # diff = fake_samples - real_samples
        # interpolated = real_samples + noise * diff

        # original paper
        random_number = tf.random.uniform((batch_size, 1))
        # print(random_number.dtype)
        # print(real_samples.dtype)
        # print(fake_samples.dtype)
        interpolated = random_number * real_samples + (1 - random_number) * fake_samples

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # Get the critic output for this interpolated image.
            pred = self.critic(interpolated)

        # Calculate the gradients w.r.t to this interpolated samples.
        grads = tape.gradient(pred, [interpolated])[0]

        # Calculate the norm of the gradients.
        # TODO verify if reduce_sum() is doing what is expected
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        grad_penalty = tf.reduce_mean((norm - 1) ** 2)
        return grad_penalty

    def train_step(self, real_samples: tf.Tensor) -> Dict[str, float]:
        batch_size = tf.shape(real_samples)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.n_critic):
            # Get latent vectors
            random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                fake_samples = self.generator(random_latent_vectors)
                fake_logits = self.critic(fake_samples)
                real_logits = self.critic(real_samples)

                critic_loss = self.critic_loss_fn(real_logits, fake_logits)
                gp = self.gradient_penalty(batch_size, real_samples, fake_samples)
                critic_loss += gp * self.lambda_

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables)
            )

        # Train the generator
        # Get latent vectors
        random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_latent_vectors)
            generated_samples_logits = self.critic(generated_samples)

            generator_loss = self.generator_loss_fn(generated_samples_logits)

        generator_grad = tape.gradient(
            generator_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_grad, self.generator.trainable_variables)
        )

        return {"critic_loss": critic_loss, "generator_loss": generator_loss}


def critic_loss(real_sample: tf.Tensor, fake_sample: tf.Tensor) -> float:
    real_loss = tf.reduce_mean(real_sample)
    fake_loss = tf.reduce_mean(fake_sample)
    return fake_loss - real_loss


def generator_loss(fake_sample: tf.Tensor) -> float:
    return -tf.reduce_mean(fake_sample)


def main(dataset: tf.data.Dataset, epochs: int) -> None:
    latent_dim = 8
    lambda_ = 10.0
    n_critic = 5
    alpha = 0.0001
    beta_1, beta_2 = 0.0, 0.9

    critic = Critic()
    # TODO get from data the input dim
    generator = Generator(input_dim=20)

    critic_optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha, beta_1=beta_1, beta_2=beta_2
    )
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha, beta_1=beta_1, beta_2=beta_2
    )

    # Get the Wasserstein GAN model
    wgan = WGAN(
        critic=critic,
        generator=generator,
        latent_dim=latent_dim,
        n_critic=n_critic,
        lambda_=lambda_,
    )

    # Compile the Wasserstein GAN model
    wgan.compile(
        critic_optimizer=critic_optimizer,
        generator_optimizer=generator_optimizer,
        critic_loss_fn=critic_loss,
        generator_loss_fn=generator_loss,
    )

    # Start training
    # TODO Load SIRR composition data
    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(monitor="generator_loss", patience=100)
    # ]

    history = wgan.fit(dataset, epochs=epochs)
    return history.history
