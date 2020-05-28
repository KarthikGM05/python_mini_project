from flask import Flask, request, render_template, session, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from wtforms import Form, FileField, validators, SubmitField
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import time
import imageio
import glob
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['UPLOAD_FOLDER'] = '.'
style_targets = {}
base_targets = {}


class ImageForm(FlaskForm):
    base = FileField("Base Image", [validators.DataRequired()], render_kw={
                     'accept': 'image/*', 'onchange': 'readURL(this, "#base-img")', 'style': 'padding: 5px;'})
    style = FileField("Style Image", [validators.DataRequired()], render_kw={
                      'accept': 'image/*', 'onchange': 'readURL(this, "#style-img")', 'style': 'padding: 5px;'})
    submit = SubmitField("Mix")


def load_img(file_path):
    max_dim = 512
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Tweak shape so the maximum dimension is 512px.
    # This is the max shape that VGG19 expects.
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    longer_dim = max(shape)
    scale = max_dim / longer_dim
    # Scale shape according to the size of the larger dim.
    rescaled = tf.cast(shape * scale, tf.int32)
    # Resize the image with rescaled shape.
    img = tf.image.resize(img, rescaled)
    # Expand the dimension as this is what the model expects.
    # [B, I, J, C]
    img = tf.expand_dims(img, axis=0)
    return img


def transfer_vgg(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)

    return model


# The style of an image can be illustrated by the correlation across different
# filters.
def gram_matrix(t):
    result = tf.einsum('bijc, bijd->bcd', t, t)
    shape = tf.shape(t)
    # num loc is the area of pixels on the image.
    num_loc = tf.cast(shape[1]*shape[2], tf.float32)
    return result/(num_loc)


class StyleBaseModel(tf.keras.models.Model):
    def __init__(self, style_layers, base_layers):
        super(StyleBaseModel, self).__init__()
        self.vgg = transfer_vgg(style_layers + base_layers)
        self.style_layers = style_layers
        self.base_layers = base_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, base_outputs = (outputs[:self.num_style_layers],
                                       outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        base_dict = {base_name: value
                     for base_name, value
                     in zip(self.base_layers, base_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'base': base_dict, 'style': style_dict}


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def mse(y, y_hat):
    return tf.reduce_mean((y - y_hat) ** 2)


def style_base_loss(outputs):
    style_outputs = outputs['style']
    base_outputs = outputs['base']
    style_loss = tf.add_n([mse(style_outputs[name], style_targets[name])
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    base_loss = tf.add_n([mse(base_outputs[name], base_targets[name])
                          for name in base_outputs.keys()])
    base_loss *= base_weight / num_base_layers
    loss = style_loss + base_loss
    return loss


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return pil.fromarray(tensor)


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = style_base_model(image)
        loss = style_base_loss(outputs)
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    optimiser.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# Break out the VGG19 network.
vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
base_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
num_base_layers = len(base_layers)
num_style_layers = len(style_layers)
style_base_model = StyleBaseModel(style_layers, base_layers)
optimiser = tf.optimizers.Adam(learning_rate=0.02)
style_weight = 1e-2
base_weight = 1e3
total_variation_weight = 30


@app.route('/')
def index():
    form = ImageForm()
    return render_template('index.html', form=form)


@app.route("/result", methods=['POST'])
def result():
    b = request.files['base']
    b.save(b.filename)
    s = request.files['style']
    s.save(s.filename)
    base = load_img(b.filename)
    style = load_img(s.filename)
    global style_targets
    global base_targets
    style_targets = style_base_model(style)['style']
    base_targets = style_base_model(base)['base']
    image = tf.Variable(base)
    epochs = 20
    steps_per_epoch = 10
    images = []
    step = 0
    for n in range(epochs):
        print(f"Epoch {n+1}: ", end='')
        for m in range(steps_per_epoch):
            step += 1
            images.append(tensor_to_image(image))
            train_step(image)
            print(".", end='')
    images.append(tensor_to_image(image))
    for i, img in enumerate(images):
        img.save(f"image-{i}.jpeg")
    anim_file = 'res.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.jpeg')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    return render_template('result.html')


@app.route('/getImage')
def getImage():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'res.gif')
