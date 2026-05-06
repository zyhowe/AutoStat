import paddle
from paddlenlp.transformers import UIE
from paddlenlp.taskflow.utils import static_mode_guard

# 加载模型
model = UIE.from_pretrained('uie-base')
model.eval()

# 保存静态图模型
static_model_path = "./uie_static"
model.save_pretrained(static_model_path, convert_to_static=True)

print(f"静态模型已保存到 {static_model_path}")