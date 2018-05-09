import os
import argparse

import numpy as np
import tensorflow as tf

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
                file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
                tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def classify_tensor(image_tensor):
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    return sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: image_tensor
    })

def printProbabilities(results):
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i], results[i])

def getPrediction(results):
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    return labels[top_k[0]]

def classify_array(nparr):
    # Load tensor from array
    t = np.reshape(nparr, (1, 299, 299, 3))

    # Run the tensor through the neural network
    results = classify_tensor(t)
    printProbabilities(results)
    return getPrediction(results)

def classify_file(file_name):
    path = file_base + file_name
    # Load tensor from image
    t = read_tensor_from_image_file(
            path,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

    # Run the tensor through the neural network
    results = classify_tensor(t)
    printProbabilities(results)
    return getPrediction(results)

file_base = "images/"
model_file = "output_graph.pb"
label_file = "output_labels.txt"
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer = "Mul"
output_layer = "final_result"
graph = load_graph(model_file)
sess = tf.Session(graph=graph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="directory where the images are stored")
    args = parser.parse_args()
    if args.images:
        file_base = args.images + "/"

    directory = os.fsencode(file_base)
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".gif") or file_name.endswith(".bmp"):
            print(file_name, " is a ", classify_file(file_name))