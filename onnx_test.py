import torch
import onnxruntime
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from utils.utils import get_max_preds
import time


def main():
    parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

    session = onnxruntime.InferenceSession("experiments/output/kd_hourglass.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape

    image = plt.imread("test.jpg")

    trans = transforms.ToTensor()

    input = trans(image).unsqueeze(0).numpy()
    end = time.time()
    output = session.run([output_name], {input_name: input})[0]
    print(time.time() - end)

    preds = get_max_preds(output)[0] * 4

    plt.figure()
    plt.imshow(image)
    plt.scatter(preds[:, 0], preds[:, 1], marker='.', color='blue', s=40)
    x, y = [], []
    for k in range(16):
        x.append([preds[k][0], preds[parent_ids[k]][0]])
        y.append([preds[k][1], preds[parent_ids[k]][1]])
    for k in range(16):
        plt.plot(x[k], y[k], color='red')
    plt.show()


if __name__ == "__main__":
    main()