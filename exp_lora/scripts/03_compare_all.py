"""
三方对比测试：自训基座 vs 自训+LoRA vs 官方模型
生成 lora_comparison.md
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
# 下面这行要根据实际 API 调整
from model.model_lora import apply_lora, load_lora

DEVICE = 'cuda'

# ============================================================
# 测试问题
# ============================================================

# 🔴 文档相关问题：你自己填 5-8 条
DOC_QUESTIONS = [
    "请介绍文档的主要内容。",
    "华中科技大学校训是什么？",
    "华中科技大学研究生最长学习年限的具体规定分哪些类别？",
    "华中科技大学研究生课程免修的外语类条件有哪些？",
    "华中科技大学研究生学业奖学金的等级与金额标准分别是什么？",
    "华中科技大学对研究生考试作弊行为的处分规定是什么？",
]

# 🔵 通用常识问题：由我提供（多样化的类别，测试综合能力和是否灾难性遗忘）
GENERAL_QUESTIONS = [
    # 基础常识
    "中国的首都是哪里？",
    # 科学常识
    "水的沸点是多少摄氏度？",
    # 数学推理
    "1+1等于几？请解释为什么。",
    # 语言创作
    "请写一句话赞美春天。",
    # 自我认知
    "请简短介绍一下你自己。",
]

ALL_QUESTIONS = [('doc', q) for q in DOC_QUESTIONS] + [('general', q) for q in GENERAL_QUESTIONS]
print(f"共 {len(ALL_QUESTIONS)} 个问题（{len(DOC_QUESTIONS)} 文档 + {len(GENERAL_QUESTIONS)} 通用）")

# ============================================================
# 加载模型
# ============================================================

print("\n" + "=" * 60)
print("步骤 1：加载 tokenizer")
tokenizer_mine = AutoTokenizer.from_pretrained('./model/')
tokenizer_official = AutoTokenizer.from_pretrained('./MiniMind2-Small-Official')

print("\n步骤 2：加载自训 SFT 基座模型（组 A）")
config = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
model_A = MiniMindForCausalLM(config)
state = torch.load('./out/full_sft_512.pth', map_location=DEVICE)
model_A.load_state_dict(state, strict=False)
model_A = model_A.eval().to(DEVICE)

print("\n步骤 3：加载自训 SFT + LoRA 模型（组 B）")
# 重新创建一个基座 → 注入 LoRA → 加载 LoRA 权重
model_B = MiniMindForCausalLM(config)
model_B.load_state_dict(torch.load('./out/full_sft_512.pth', map_location=DEVICE), strict=False)
apply_lora(model_B)  # ⚠️ 这行 API 调用可能要调整
model_B = model_B.to(DEVICE)
load_lora(model_B, './exp_lora/models/from_mine_512.pth')  # ⚠️ 这行 API 调用可能要调整
model_B = model_B.eval()

print("\n步骤 4：加载官方模型（组 C）")
model_C = AutoModelForCausalLM.from_pretrained(
    './MiniMind2-Small-Official',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(DEVICE).eval()

print("\n✅ 三个模型都加载完成")

# ============================================================
# 生成函数
# ============================================================

def generate(model, tokenizer, prompt, max_new=400):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new,
            temperature=0.85,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# ============================================================
# 逐个测试
# ============================================================

results = []
for i, (qtype, q) in enumerate(ALL_QUESTIONS, 1):
    print(f"\n{'=' * 60}")
    print(f"[{qtype.upper()}] 测试 {i}/{len(ALL_QUESTIONS)}: {q}")
    print('-' * 60)

    a = generate(model_A, tokenizer_mine, q)
    print(f"[A 自训基座] {a[:100]}...")

    b = generate(model_B, tokenizer_mine, q)
    print(f"[B 自训+LoRA] {b[:100]}...")

    c = generate(model_C, tokenizer_official, q)
    print(f"[C 官方模型] {c[:100]}...")

    results.append((qtype, q, a, b, c))

# ============================================================
# 生成 Markdown 报告
# ============================================================

output_path = './exp_lora/results/lora_comparison.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("# LoRA 微调效果三方对比测试\n\n")
    f.write("## 实验设计\n\n")
    f.write("| 组别 | 模型 | 说明 |\n")
    f.write("|------|------|------|\n")
    f.write("| A | 自训 SFT 基座 | 未做 LoRA，作为微调前基线 |\n")
    f.write("| B | 自训 SFT + LoRA | 自训模型 + 文档微调，评估 LoRA 增益 |\n")
    f.write("| C | 官方 MiniMind2-Small | 强基座参照（未做文档微调）|\n\n")
    f.write("## 文档相关问题（考察文档知识掌握）\n\n")
    f.write("| # | 问题 | A: 自训基座 | B: 自训+LoRA | C: 官方模型 |\n")
    f.write("|---|------|-----------|-------------|------------|\n")
    doc_idx = 0
    for qtype, q, a, b, c in results:
        if qtype == 'doc':
            doc_idx += 1
            a_md = a.replace('\n', '<br>').replace('|', '\\|')
            b_md = b.replace('\n', '<br>').replace('|', '\\|')
            c_md = c.replace('\n', '<br>').replace('|', '\\|')
            f.write(f"| {doc_idx} | {q} | {a_md} | {b_md} | {c_md} |\n")

    f.write("\n## 通用常识问题（考察是否灾难性遗忘）\n\n")
    f.write("| # | 问题 | A: 自训基座 | B: 自训+LoRA | C: 官方模型 |\n")
    f.write("|---|------|-----------|-------------|------------|\n")
    gen_idx = 0
    for qtype, q, a, b, c in results:
        if qtype == 'general':
            gen_idx += 1
            a_md = a.replace('\n', '<br>').replace('|', '\\|')
            b_md = b.replace('\n', '<br>').replace('|', '\\|')
            c_md = c.replace('\n', '<br>').replace('|', '\\|')
            f.write(f"| {gen_idx} | {q} | {a_md} | {b_md} | {c_md} |\n")

    f.write("\n## 观察要点\n\n")
    f.write("1. **文档问题 A vs B**：若 B 明显优于 A，说明 LoRA 成功学到了文档内容\n")
    f.write("2. **文档问题 B vs C**：若 B 优于 C，说明 LoRA 弥补了基座数据量的不足\n")
    f.write("3. **通用问题 A vs B**：若 B 大幅退化，说明出现了灾难性遗忘\n")
    f.write("4. **通用问题 B vs C**：官方模型预期在通用问题上更强（数据量优势）\n")

print(f"\n✅ 三方对比完成！结果保存到 {output_path}")