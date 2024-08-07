import os
os.chdir(f"/home/xlab-app-center")
os.system(f"git clone -b totoro3 https://github.com/camenduru/ComfyUI /home/xlab-app-center/TotoroUI")
os.chdir(f"/home/xlab-app-center/TotoroUI")
os.system(f"git lfs install")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://download.openxlab.org.cn/models/camenduru/flux/weight/flux1-dev.sft -d /home/xlab-app-center/TotoroUI/models/unet -o flux1-dev.sft")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://download.openxlab.org.cn/models/camenduru/flux/weight/ae.sft -d /home/xlab-app-center/TotoroUI/models/vae -o ae.sft")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://download.openxlab.org.cn/models/camenduru/flux/weight/clip_l.safetensors -d /home/xlab-app-center/TotoroUI/models/clip -o clip_l.safetensors")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://download.openxlab.org.cn/models/camenduru/flux/weight/t5xxl_fp16.safetensors -d /home/xlab-app-center/TotoroUI/models/clip -o t5xxl_fp16.safetensors")
os.system(f"python launch.py")