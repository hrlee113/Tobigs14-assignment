import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_datasets as tfds
import keras
from keras.models import *
from keras.layers import *
from tensorflow.keras import initializers

class Generator(tf.keras.Model):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.noise_size = 100
    
    def __call__(self, training=True):
        
        model = Sequential()
        
        self.G.add(Dense(1024, input_shape=(self.noise_size,)))
        self.G.add(BatchNormalization(momentum = 0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((1,1,1024))
        
        self.G.add(Conv2D(512, kernel_initializer=initializers.RandomNormal(0, 0.02), kernel_size=5, strides=(2,2)))
        self.G.add(BatchNormalization(momentum = 0.9))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
        
        self.G.add(Conv2D(256, kernel_initializer=RandomNormal(mean=0, stddev=0.02), kernel_size=5, strides=(2,2)))
        self.G.add(BatchNormalization(momentum = 0.9))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
        
        self.G.add(Conv2D(128, kernel_initializer=RandomNormal(mean=0, stddev=0.02), kernel_size=5, strides=(2,2)))
        self.G.add(BatchNormalization(momentum = 0.9))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
        
        self.G.add(Conv2D(1, kernel_initializer=RandomNormal(mean=0, stddev=0.02), kernel_size=5, strides=(2,2)))
        self.G.add(Activation('tanh'))
        self.G.add(UpSampling2D())
        
        return self.G
    

class Discriminator(tf.keras.Model):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.input_size = (64, 64, 1)
    
    def __call__(self):
        
        self.D = Sequential()
        
        self.D.add(Conv2D(128, kernel_initializer=RandomNormal(mean=0, stddev=0.02),  kernel_size=5, strides=(2,2)))
        self.D.add(BatchNormalization(momentum = 0.9))
        self.D.add(LeakyReLU(0.2))
        
        self.D.add(Conv2D(256, kernel_initializer=RandomNormal(mean=0, stddev=0.02), kernel_size=5, strides=(2,2)))
        self.D.add(BatchNormalization(momentum = 0.9))
        self.D.add(LeakyReLU(0.2))
        
        self.D.add(Conv2D(512, kernel_initializer=RandomNormal(mean=0, stddev=0.02), kernel_size=5, strides=(2,2)))
        self.D.add(BatchNormalization(momentum = 0.9))
        self.D.add(LeakyReLU(0.2))
        
        self.D.add(Conv2D(1024, kernel_initializer=RandomNormal(mean=0, stddev=0.02), kernel_size=5, strides=(2,2)))
        self.D.add(BatchNormalization(momentum = 0.9))
        self.D.add(LeakyReLU(0.2))
        
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        
        return self.D
        

def discriminator_loss(loss_object, real_output, fake_output):
    #here = tf.ones_like(????) or tf.zeros_like(????)  -> tf.zeros_like와 tf.ones_like에서 선택하고 (???)채워주세요
    real_loss = loss_object(tf.ones_like((batch_size,1)), real_output)
    fake_loss = loss_object(tf.ones_like((batch_size,1)), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like((batch_size,1)), fake_output)

def normalize(x):
    image = tf.cast(x['image'], tf.float32)
    image = (image / 127.5) - 1
    return image


def save_imgs(epoch, generator, noise):
    gen_imgs = generator(noise, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

def train():
    data, info = tfds.load("mnist", with_info=True, data_dir='/data/tensorflow_datasets')
    train_data = data['train']

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    latent_dim = 100
    epochs = 2
    batch_size = 10000
    buffer_size = 6000
    save_interval = 1

    generator = Generator()
    discriminator = Discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.999) #beta_2 : default
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.999) #beta_2 : default

    train_dataset = train_data.map(normalize).shuffle(buffer_size).batch(batch_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy() #진짜는 1, 가짜는 0

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim]) #z

        with tf.GradientTape(persistent=True) as tape:
            generated_images = generator(noise)

            real_output = discriminator(images)
            generated_output = discriminator(generated_images)

            gen_loss = generator_loss(cross_entropy, generated_images)
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output)

        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for images in train_dataset:
            gen_loss, disc_loss = train_step(images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss / batch_size, total_disc_loss / batch_size))
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()