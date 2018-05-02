import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    print(model_data.keys())
    print(opt.arch, model_data['arch'])
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    input_files = input_files[::-1]
    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    #if os.path.exists('tmp'):
    #    subprocess.call('rm -rf tmp', shell=True)

    for input_file in input_files:
        outputs = []
        if 'v_' == input_file[:2]:
            input_file = input_file[2:]
        video_path = os.path.join(opt.video_root, input_file)
        #name = os.path.basename(input_file).split('.')[0]
        name = input_file.replace('.mp4', '')
        if 'v_' == name[:2]:
            name = name[2:]
        output_filename = os.path.join('/checkpoint02/spandanagella/dataset/activitynet/3DResNet/%s_%s' %(opt.model_name, opt.model_depth), name+'.json')
        print video_path
        if os.path.exists(output_filename) and os.path.getsize(output_filename)>1000:
            continue
        #print('Adding', video_path)
        if os.path.exists(video_path):
            print(video_path)
            print(input_file)
            #print(video_name)
            video_name = '/checkpoint02/spandanagella/dataset/activitynet/3DResNet/tmp_files/%s_images' %(name)
            #sys.exit(0)
            subprocess.call('mkdir -p %s' %(video_name), shell=True)
            #subprocess.call('ffmpeg -i {} %s/image_%05d.jpg'.format(video_path, video_name),
            #                shell=True)
            command = 'ffmpeg -i '+video_path+ ' %s/image_' %(video_name)+'%05d.jpg'
            print(command)
            #sys.exit(0)
            os.system(command)

            print('Number of images', len(os.listdir(video_name)))
            #sys.exit(0)

            #for filename in os.listdir('./tmp'):
            #    print(filename)

            result = classify_video(video_name, input_file, class_names, model, opt)
            outputs.append(result)

            #subprocess.call('rm -rf tmp', shell=True)
            #if os.path.exists(video_name):
            #    subprocess.call('rm -rf %s' %(video_name), shell=True)

            output_filename = os.path.join('/checkpoint02/spandanagella/dataset/activitynet/3DResNet/%s_%s' %(opt.model_name, opt.model_depth), name+'.json')
            if len(outputs[0]['clips']) >0:
                with open(output_filename, 'w') as f:
                    json.dump(outputs, f)
            print output_filename
            #sys.exit(0)
        else:
            print('{} does not exist'.format(input_file))
        #break

