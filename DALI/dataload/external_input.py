# Ref: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html#Define-the-Data-Source

import timeit
import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import matplotlib.pyplot as plt
import cupy as cp
import imageio

batch_size = 1


class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.images_dir = "../data/images/"
        self.batch_size = batch_size
        with open(self.images_dir + "file_list.txt", "r") as f:
            self.files = [line.rstrip() for line in f if line != ""]
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i].split(" ")
            f = open(self.images_dir + jpeg_filename, "rb")
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)


class ExternalInputGpuIterator(object):
    def __init__(self, batch_size):
        self.images_dir = "../data/images/"
        self.batch_size = batch_size
        with open(self.images_dir + "file_list.txt", "r") as f:
            self.files = [line.rstrip() for line in f if line != ""]
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i].split(" ")
            # 通过定义的图片路径，将图片完整读入 CPU 内存。返回值是 numpy array 数据类型
            im = imageio.imread(self.images_dir + jpeg_filename)

            # 将 numpy array 转换为 cupy array。转换完的数据将会存储在 GPU 内存。
            im = cp.asarray(im)

            # 简单预处理乘以 0.6，调整亮度对比度之类。
            im = im * 0.6
            batch.append(im.astype(cp.uint8))
            labels.append(cp.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)


def interact_with_cpu():
    eii = ExternalInputIterator(batch_size)
    pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    with pipe:
        jpegs, labels = fn.external_source(
            source=eii, num_outputs=2, dtype=types.UINT8
        )
        decode = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        enhance = fn.brightness_contrast(decode, contrast=2)
        pipe.set_outputs(enhance, labels)

    pipe.build()

    execution_time_cpu_pipe_run = timeit.timeit(pipe.run, number=1)
    print("CPU pipe_cpu.run Execution time: {:.2f}".format(execution_time_cpu_pipe_run))

    pipe_out = pipe.run()
    # print("pipe_out len: {}".format(len(pipe_out)))
    batch_cpu, labels_cpu = pipe_out[0].as_cpu(), pipe_out[1]

    img = batch_cpu.at(0)
    # print(img.shape)
    # print(labels_cpu.at(0))
    plt.axis("off")
    plt.imshow(img)
    plt.imsave(eii.images_dir + "test-cpu.jpg", img)


def interact_with_gpu():
    eii_gpu = ExternalInputGpuIterator(batch_size)
    # print(type(next(iter(eii_gpu))[0][0]))
    pipe_gpu = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    with pipe_gpu:
        images, labels = fn.external_source(
            source=eii_gpu, num_outputs=2, device="gpu", dtype=types.UINT8
        )
        """这里注释了解码部分"""
        # decode = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        enhance = fn.brightness_contrast(images, contrast=2)
        pipe_gpu.set_outputs(enhance, labels)

    pipe_gpu.build()

    execution_time_gpu_pipe_run = timeit.timeit(pipe_gpu.run, number=1)
    print("GPU pipe_gpu.run Execution time: {:.2f}".format(execution_time_gpu_pipe_run))

    pipe_out_gpu = pipe_gpu.run()
    batch_gpu = pipe_out_gpu[0].as_cpu()
    labels_gpu = pipe_out_gpu[1].as_cpu()

    img = batch_gpu.at(0)
    # print(img.shape)
    # print(labels_gpu.at(0))
    plt.axis("off")
    plt.imsave(eii_gpu.images_dir + "test-gpu.jpg", img)


def main():
    execution_time_gpu = timeit.timeit(interact_with_gpu, number=1)
    execution_time_cpu = timeit.timeit(interact_with_cpu, number=1)

    print("GPU Execution time: {:.2f}".format(execution_time_gpu))
    print("CPU Execution time: {:.2f}".format(execution_time_cpu))

    """
    GPU pipe_gpu.run Execution time: 0.09
    CPU pipe_cpu.run Execution time: 0.01
    GPU Execution time: 0.34
    CPU Execution time: 0.09
                                                        
    Execution Process:                            
    Pipeline(imageURL, decoder) -> pipeline.run() -> next_batch()                   -> decoders() -> enhance() -> pipeline.out()
                                                                  -> imageio.imread                                                    
                                                    |Pipeline > > > > > > > > > > > > > > > > > > > > > > > > > |
    """


if __name__ == '__main__':
    main()
