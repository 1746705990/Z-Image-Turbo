import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 基础配置
MODEL_PATH = os.path.join(BASE_DIR, "Z-Image-Turbo")
OUTPUT_DIR = os.path.join(BASE_DIR, "img")
HTTP_PORT = 5000
DEFAULT_NEG = "ugly, deformed, noisy, blurry, low contrast, text, watermark, bad anatomy, bad hands, low quality"

# ================= 安全边界配置 =================
MAX_RESOLUTION = 2048       # 最大分辨率，防止显存OOM溢出 (拒绝服务攻击)
MIN_RESOLUTION = 256        # 最小分辨率
MAX_STEPS = 20              # 最大推理步数，防止长时间占用GPU
MAX_PROMPT_LENGTH = 1000    # 提示词最大长度，防止超大文本攻击
MAX_QUEUE_SIZE = 100        # 最大队列积压量，防止内存耗尽