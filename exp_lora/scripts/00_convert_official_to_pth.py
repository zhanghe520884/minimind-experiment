"""把官方 safetensors 权重转成 MiniMind 训练脚本能读的 .pth 格式"""
import torch
from transformers import AutoModelForCausalLM

DEVICE = 'cuda'
OUTPUT_PATH = './out/official_512.pth'

print("加载官方模型...")
model = AutoModelForCausalLM.from_pretrained(
    './MiniMind2-Small-Official',
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

print("提取权重字典...")
state_dict = model.state_dict()
print(f"权重 key 总数: {len(state_dict)}")
print(f"示例 keys（前 5 个）:")
for k in list(state_dict.keys())[:5]:
    print(f"  {k}  shape={tuple(state_dict[k].shape)}")

# 转成半精度保存（和训练脚本保存格式一致）
state_dict_half = {k: v.half().cpu() for k, v in state_dict.items()}

print(f"\n保存到 {OUTPUT_PATH} ...")
torch.save(state_dict_half, OUTPUT_PATH)

print(f"✅ 转换完成！文件大小应该在 50-60 MB")