import torch
import os
import json
import time
import signal
from datetime import datetime
from modelscope import ZImagePipeline
import config

def gpu_worker(rank, model_path, output_dir, queue):
    # 忽略主进程的 Ctrl+C，由主进程统一管控
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    device = f"cuda:{rank}"
    try:
        torch.cuda.set_device(device)
        print(f"[GPU {rank}] 正在加载模型...")
        torch.cuda.empty_cache()
        pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
        print(f"[GPU {rank}] 就绪")
    except Exception as e:
        print(f"[GPU {rank}] ❌ 初始化失败: {e}")
        return

    while True:
        try:
            task = queue.get()
            p, w, h, s = task['p'], task['w'], task['h'], task['s']
            steps, cfg = task['steps'], task['cfg']
            
            print(f"[GPU {rank}] 绘图: {w}x{h} | S:{s} | CFG:{cfg}")
            g = torch.Generator(device).manual_seed(s)
            
            with torch.inference_mode():
                imgs = pipe(
                    prompt=p,
                    negative_prompt=config.DEFAULT_NEG,
                    height=h, width=w,
                    num_inference_steps=steps, 
                    guidance_scale=cfg,
                    num_images_per_prompt=4,
                    generator=g
                ).images
                
                for i, img in enumerate(imgs):
                    ts = datetime.now().strftime("%H%M%S")
                    fn = f"{w}x{h}_{s}_{ts}_g{rank}_{i}.png"
                    
                    # 使用临时文件写入，避免前端读到残缺的图片
                    tmp_img_path = os.path.join(output_dir, f"tmp_{fn}")
                    tmp_json_path = os.path.join(output_dir, f"tmp_{fn}.json")
                    
                    img.save(tmp_img_path)
                    meta = {
                        "prompt": p, "width": w, "height": h, "seed": s,
                        "steps": steps, "cfg": cfg, "timestamp": ts, "gpu": rank
                    }
                    with open(tmp_json_path, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, ensure_ascii=False)
                        
                    # 写入完成后重命名 (原子操作)
                    os.rename(tmp_img_path, os.path.join(output_dir, fn))
                    os.rename(tmp_json_path, os.path.join(output_dir, fn + ".json"))
            
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[GPU {rank}] 错误: {e}")
            time.sleep(1)