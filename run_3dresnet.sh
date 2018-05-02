
CUDA_VISIBLE_DEVICES=0 python main.py --input ./video_names.txt --video_root ./test_videos --output ./output.json --model ./pretrained_models/resnext-101-kinetics.pth --mode feature --model_name resnext --model_depth 101 --resnet_shortcut B
