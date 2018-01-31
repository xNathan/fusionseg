# coding: utf-8

from __future__ import print_function

import os
import subprocess

caffe_bin = '/home/nathan/own/caffe/deeplab-public-ver2/distribute/bin/caffe.bin'
caffe_bin = '/home/nathan/own/caffe/deeplab-public-ver2-bitbucket/distribute/bin/caffe.bin'
gpu_device = 0

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data')

# 文件夹名称 - 模型名称
model_type_map = {
    'flow': 'motion',
    'image': 'appearance'
}
# 裁减的矩形区域大小
# 过大会导致GPU显存不足
image_size = 360

for root, dirs, files in os.walk(data_dir):
    if dirs:
        continue
    # 到达最深层
    # root父目录即为每个video的根目录
    image_type = root.split(os.path.sep)[-1]
    model_type = model_type_map[image_type]

    video_dir = os.path.join(root, os.path.pardir)
    video_dir = os.path.abspath(video_dir)  # 转换为绝对路径

    # 生成路径txt文件
    input_list_file = os.path.join(
        video_dir, '{}_image_list.txt'.format(model_type))
    output_list_file = os.path.join(
        video_dir, '{}_output_list.txt'.format(model_type))

    input_file = open(input_list_file, 'w')
    output_file = open(output_list_file, 'w')
    image_list = []
    for filename in files:
        if filename.lower().endswith(('.jpg', '.png', '.gif', '.jpeg', '.bmp')):
            image_list.append(filename)
    image_list.sort()

    for img in image_list:
        input_file.write('/{}\n'.format(img))
        img_name = os.path.splitext(img)[0]
        output_file.write('{}\n'.format(img_name))

    input_file.close()
    output_file.close()

    # 生成protobuf文件
    model_dir = os.path.join(base_dir, model_type)
    template_path = os.path.join(
        model_dir, model_type + '_stream_template.prototxt')
    template_file = open(template_path, 'r')
    template_data = template_file.readlines()

    test_proto_path = os.path.join(model_dir, model_type + '_stream.prototxt')
    test_proto_file = open(test_proto_path, 'w')

    tokens = {}
    tokens['${IMAGE_SIZE}'] = 'crop_size: {}'.format(image_size)
    tokens['${IMAGE_DIR}'] = 'root_folder: "{}"'.format(root)
    tokens['${OUTPUT_DIR}'] = 'prefix: "{}/"'.format(root)  # 最后要加上/

    tokens['${IMAGE_LIST}'] = 'source: "{}"'.format(input_list_file)
    tokens['${IMAGE_OUTPUT_LIST}'] = 'source: "{}"'.format(output_list_file)

    for line in template_data:
        line = line.rstrip()
        for key in tokens:
            if line.find(key) != -1:
                line = '\t' + tokens[key]
                break
        test_proto_file.write(line + '\n')
    template_file.close()
    test_proto_file.close()

    model_weight_path = os.path.join(
        model_dir, model_type + '_stream.caffemodel')
    # cmd = ' '.join([caffe_bin, 'test', '--model='+test_proto_path, '--weights='+model_weight_path,
    #                 '--gpu='+str(gpu_device), '--iterations='+str(len(image_list))])
    # print(cmd)
    args = [caffe_bin, 'test', '--model', test_proto_path, '--weights',
            model_weight_path, '--iterations', str(len(image_list))]
    if gpu_device is not None:
        args.extend(['--gpu', str(gpu_device)])
    p = subprocess.Popen(args)
    p.wait()

