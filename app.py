import os
import sys
import signal
import random
import json
import torch.multiprocessing as mp
from flask import Flask, render_template, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename
import logging

import config
from worker import gpu_worker

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

processes = []
shared_queue = None

# ================= 辅助安全函数 =================
def is_safe_path(basedir, path, follow_symlinks=True):
    # 防范文件路径穿越 (Path Traversal) 注入攻击
    if follow_symlinks:
        matchpath = os.path.abspath(path)
    else:
        matchpath = os.path.realpath(path)
    return basedir == os.path.commonpath((basedir, matchpath))

# ================= API 路由 =================
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    if not os.path.exists(config.OUTPUT_DIR): 
        return jsonify({"queue": 0, "total": 0})
    count = len([n for n in os.listdir(config.OUTPUT_DIR) if n.endswith('.png') and not n.startswith('tmp_')])
    q_size = shared_queue.qsize() if shared_queue else 0
    return jsonify({"queue": q_size, "total": count})

@app.route('/api/images')
def get_images():
    if not os.path.exists(config.OUTPUT_DIR): return jsonify([])
    files = [f for f in os.listdir(config.OUTPUT_DIR) if f.endswith('.png') and not f.startswith('tmp_')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(config.OUTPUT_DIR, x)), reverse=True)
    return jsonify(files[:60])

@app.route('/api/meta/<filename>') # 移除 path: 前缀，严格限制只能是单级文件名
def get_meta(filename):
    # 安全校验：防止 ../../../etc/passwd 这种注入
    safe_filename = secure_filename(filename)
    if not safe_filename or safe_filename != filename:
        return jsonify({"error": "非法的文件名"}), 400
        
    json_path = os.path.join(config.OUTPUT_DIR, safe_filename + ".json")
    
    # 再次验证解析后的路径是否在输出目录内
    if not is_safe_path(config.OUTPUT_DIR, json_path):
        return jsonify({"error": "越权访问被拒绝"}), 403

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f: 
                return jsonify(json.load(f))
        except Exception: 
            pass
            
    parts = safe_filename.split('_')
    return jsonify({
        "prompt": "元数据丢失",
        "width": parts[0].split('x')[0] if 'x' in parts[0] else "?",
        "height": parts[0].split('x')[1] if 'x' in parts[0] else "?",
        "seed": parts[1] if len(parts)>1 else "?",
        "steps": "?"
    })

@app.route('/img/<filename>')
def img(filename): 
    # Flask 自带的 send_from_directory 已包含基础防穿越保护
    # 但我们增加 secure_filename 确保绝对安全
    safe_filename = secure_filename(filename)
    return send_from_directory(config.OUTPUT_DIR, safe_filename)

@app.route('/api/generate', methods=['POST'])
def generate_api():
    if shared_queue.full():
        return jsonify({"error": "服务器繁忙，队列已满，请稍后再试"}), 429

    d = request.json or {}
    
    # 1. 拦截超长恶意 prompt，防止内存/显存溢出
    raw_prompt = str(d.get('prompt', ''))[:config.MAX_PROMPT_LENGTH].strip()
    if not raw_prompt:
        return jsonify({"error": "提示词不能为空"}), 400

    # 2. 强校验分辨率：限制最大值、最小值，并强制确保是 8 的倍数
    try:
        w = max(config.MIN_RESOLUTION, min(int(d.get('width', 1024)), config.MAX_RESOLUTION))
        h = max(config.MIN_RESOLUTION, min(int(d.get('height', 1024)), config.MAX_RESOLUTION))
        w = w - (w % 8)
        h = h - (h % 8)
        
        steps = max(1, min(int(d.get('steps', 4)), config.MAX_STEPS))
        cfg = max(0.0, min(float(d.get('cfg', 0.0)), 20.0))
        
        frontend_seed = int(d.get('seed', -1))
        final_seed = frontend_seed if frontend_seed != -1 else random.randint(0, 2**32 - 1)
    except ValueError:
        return jsonify({"error": "非法的数据类型"}), 400

    task = {
        'p': raw_prompt, 
        'w': w, 'h': h, 
        's': final_seed,
        'steps': steps, 
        'cfg': cfg
    }
    shared_queue.put(task)
    return jsonify({"status": "ok"})

def signal_handler(sig, frame):
    print("\n[系统] 正在关闭服务...")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
    sys.exit(0)

# ================= 启动逻辑 =================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    mp.set_start_method('spawn', force=True)
    
    if not os.path.exists(config.OUTPUT_DIR): 
        os.makedirs(config.OUTPUT_DIR)
        
    manager = mp.Manager()
    # 限制队列最大长度，防止被恶意请求打满内存
    shared_queue = manager.Queue(maxsize=config.MAX_QUEUE_SIZE) 
    
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"[系统] 启动中... (检测到 {gpu_count} GPU)")
    
    for r in range(gpu_count):
        p = mp.Process(target=gpu_worker, args=(r, config.MODEL_PATH, config.OUTPUT_DIR, shared_queue))
        p.daemon = True 
        p.start()
        processes.append(p)
        
    app.run(host='0.0.0.0', port=config.HTTP_PORT, threaded=True, use_reloader=False)