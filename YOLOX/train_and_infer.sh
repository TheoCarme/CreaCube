#!/bin/sh

python tools/train.py -f .\exps\custom\yolox_s.py -d 1 -b 8 --fp16 -o -c .\weights\yolox_s.pth

python tools/train.py -f .\exps\custom\yolox_m.py -d 1 -b 8 --fp16 -o -c .\weights\yolox_m.pth
 
python .\tools\demo.py video -n yolox-s -c .\YOLOX_outputs\yolox_s\best_ckpt.pth --path D:\Data\Videos\contrasted_scenevideo.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu -f .\exps\custom\yolox_s.py
python .\tools\demo.py video -n yolox-s -c .\YOLOX_outputs\yolox_s\best_ckpt.pth --path D:\Data\Videos\p1242+p1245.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu -f .\exps\custom\yolox_s.py
python .\tools\demo.py video -n yolox-s -c .\YOLOX_outputs\yolox_s\best_ckpt.pth --path D:\Data\Videos\p657_p658.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu -f .\exps\custom\yolox_s.py
python .\tools\demo.py video -n yolox-s -c .\YOLOX_outputs\yolox_s\best_ckpt.pth --path D:\Data\Videos\p705_ind_Biffoscheim_final.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu -f .\exps\custom\yolox_s.py
python .\tools\demo.py video -n yolox-s -c .\YOLOX_outputs\yolox_s\best_ckpt.pth --path D:\Data\Videos\p659.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu -f .\exps\custom\yolox_s.py


RuntimeError: CUDA out of memory. Tried to allocate 2.00 MiB (GPU 0; 8.00 GiB total capacity; 203.44 MiB already allocated; 0 bytes free; 6.44 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF