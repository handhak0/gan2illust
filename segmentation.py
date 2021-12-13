

# ------------------------------------------------------------------------------------------------------
# Pix2Pix 실행

def pix2pix(pred,filename) :
    # import torch
    import os
    import tensorflow as tf
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt

    # ---------------------------------------------------
    # 함수선언

    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Generator():
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(128, 4),  # (batch_size, 64, 64, 128)
            downsample(256, 4),  # (batch_size, 32, 32, 256)
            downsample(512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 4),  # (batch_size, 16, 16, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)



    def Discriminator():
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def load(image_file):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        # Split each image tensor into two tensors:
        # - one with a real building facade image
        # - one with an architecture label image
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        # Convert both images to float32 tensors
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def resize(input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    # Normalizing the images to [-1, 1]
    def normalize(input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image


    def load_image_test(image_file):
        input_image, real_image = load(image_file)
        input_image, real_image = resize(input_image, real_image,
                                         IMG_HEIGHT, IMG_WIDTH)
        input_image, real_image = normalize(input_image, real_image)

        return input_image, real_image

    def generate_images(model, test_input, tar, num_):
        prediction = model(test_input, training=True)
        plt.imshow(prediction[0] * 0.5 + 0.5)
        plt.axis('off')

        save_path = 'C:/Users/sonso/Desktop/Git/멀티캠퍼스/04.FinalProject/손학영/static/output/'
        plt.savefig(save_path + filename, dpi=150, bbox_inches='tight', pad_inches=0)

        # extension = os.path.splitext(save_path)[1]
        #
        # final, encoded_img = cv2.imencode(extension, pixResult)
        #
        # if final:
        #     with open(save_path, mode='w+b') as f:
        #         encoded_img.tofile(f)



    # ---------------------------------------------------
    # 변수 선언
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    OUTPUT_CHANNELS = 3

    generator = Generator()
    tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

    discriminator = Discriminator()
    tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = 'C:/Users/sonso/Desktop/Git/멀티캠퍼스/04.FinalProject/손학영/pix2pixckpt' ## Checkpoint Path
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



    # ---------------------------------------------------
    # 스타일 이미지 로드

    target_path = 'C:/Users/sonso/Desktop/Git/멀티캠퍼스/04.FinalProject/손학영/static/Style/Style1.png'

    stream = open(target_path.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    target = cv2.imdecode(numpyArray, cv2.IMREAD_COLOR)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    target = cv2.resize(target, dsize=(256, 256))  # 크기 변환이 필요시

    img_merge = np.hstack((target, pred))
    img_merge = Image.fromarray(img_merge)

    plt.imshow(img_merge)
    plt.axis('off')
    plt.savefig(('./test_1.jpg'), dpi=250, bbox_inches='tight', pad_inches=0)


    BATCH_SIZE = 1
    m_img = tf.data.Dataset.list_files('./test_1.jpg')
    m_img = m_img.map(load_image_test)
    m_img = m_img.batch(BATCH_SIZE)


    ## Generate Image
    for inp, tar in m_img:
        generate_images(generator, inp, tar, 0)



# ------------------------------------------------------------------------------------------------------------------
# Segmentation 실행

def segmentation(filename):
    from tensorflow.keras import Input
    from model import u_net
    import cv2

    import tensorflow as tf
    import numpy as np


    # ---------------------------------------------------
    # 함수선언
    input_img = Input(shape=(256, 256, 3), name='img')
    model = u_net.get_u_net(input_img, num_classes=19)
    model.summary()
    model.load_weights('C:/Users/sonso/Desktop/Git/멀티캠퍼스/04.FinalProject/손학영/Face_segmentation/1125_1250-unet.h5')

    color_list = [[0, 0, 0], [204, 0, 0], [255, 140, 26], [204, 204, 0],
                  [51, 51, 255], [204, 0, 204], [0, 255, 255],
                  [255, 204, 204], [102, 51, 0], [255, 0, 0],
                  [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
                  [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    def change_3ch(prediction, image):
        """
        Change image chanel to 3 chanel

        :param prediction: predicted output of the model.
        :type prediction: array
        :param mask: true masks of the images.
        :type mask: array
        :param image: original image.
        :type image: array

        """
        # tmp_image = (image * 255.0).astype(np.uint8)
        im_base = np.zeros((256, 256, 3), dtype=np.uint8)
        for idx, color in enumerate(color_list):
            im_base[prediction == idx] = color
        return im_base

    # ---------------------------------------------------
    # 유저 이미지 로드
    img_file = 'C:/Users/sonso/Desktop/Git/멀티캠퍼스/04.FinalProject/손학영/static/input/' + filename

    stream = open(img_file.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    img = cv2.imdecode(numpyArray, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = tf.image.random_brightness(img, max_delta=0.5)

    dst = cv2.resize(np.float32(img), dsize=(256, 256))  # , interpolation=cv2.INTER_AREA)
    dst = tf.stack([dst])

    # ---------------------------------------------------
    # 모델 수행
    predictions = model.predict(dst)
    predictions = np.argmax(predictions, axis=-1)

    pred = change_3ch(predictions[0], dst[0].numpy())
    # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

    pix2pix(pred, filename)





# cv2.imwrite(save_path, pixResult)

# if __name__ == "__main__" :
#     pass


# segmentation()