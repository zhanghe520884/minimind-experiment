"""
三方对比测试: 自训 64M / 自训 64M+LoRA / 官方 minimind-3
完全复用官方 eval_llm.py 的模型加载与生成逻辑
输出: comparison_three_way.md
"""
import os
import time
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

# ============ 配置 ============
DEVICE = 'cuda:0'
HIDDEN = 768
LAYERS = 8

OUTPUT_MD = './comparison_three_way.md'

# 测试问题集 (照搬 eval_llm.py 里的, 通用题 + 华科文档题)
PROMPTS = [
    # === 通用能力 ===
    '你有什么特长?',
    '请用Python写一个计算斐波那契数列的函数',
    '解释一下"光合作用"的基本过程',
    '如果明天下雨, 我应该如何出门',
    '1+2等于几?',
    '比较一下猫和狗作为宠物的优缺点',
    '解释什么是机器学习',
    '推荐一些中国的美食',
    # === 华科文档专有知识 (LoRA 微调内容) ===
    "华中科技大学校训是什么?",
    "华中科技大学研究生课程免修的外语类条件有哪些?",
    "华中科技大学研究生学业奖学金的等级与金额标准分别是什么?",
    "华中科技大学研究生课程任课教师应具备什么职称条件?",
    "华中科技大学研究生国家奖学金硕士奖励标准是多少?",
]

# 区分"通用题"和"文档题"的分界(前 8 题通用, 后 5 题文档)
DOC_START_IDX = 8


# ============ 模型加载 (照搬 eval_llm.py 的 init_model 逻辑) ============
def load_self_trained(weight, lora_weight=None):
    """加载自训 MiniMind 模型(可选叠加 LoRA)"""
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
    ))
    ckp = f'./out/{weight}_{HIDDEN}.pth'
    model.load_state_dict(torch.load(ckp, map_location=DEVICE), strict=True)
    if lora_weight:
        apply_lora(model)
        load_lora(model, f'./out/{lora_weight}_{HIDDEN}.pth')
    return model.half().eval().to(DEVICE)


def load_official(path):
    """加载 transformers 格式的官方模型"""
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    return model.half().eval().to(DEVICE)


print("=" * 60)
print("加载 tokenizer...")
tokenizer_mine = AutoTokenizer.from_pretrained('./model')
tokenizer_official = AutoTokenizer.from_pretrained('./minimind-3-Official')

print("\n[模型 A] 加载自训 64M (full_sft_768.pth)...")
model_A = load_self_trained(weight='full_sft')
get_model_params(model_A, model_A.config)

print("\n[模型 B] 加载自训 64M + LoRA (from_mine_768.pth)...")
model_B = load_self_trained(weight='full_sft', lora_weight='from_mine')
get_model_params(model_B, model_B.config)

print("\n[模型 C] 加载官方 minimind-3...")
model_C = load_official('./minimind-3-Official')
get_model_params(model_C, model_C.config)

print("\n✅ 三个模型加载完成\n")


# ============ 生成函数 (照搬 eval_llm.py 的逻辑, 只调整生成参数防重复) ============
def generate(model, tokenizer, prompt):
    setup_seed(random.randint(0, 31415926))
    conversation = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(DEVICE)
    
    st = time.time()
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=400,              # 限制长度避免长重复
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.85,
        temperature=0.6,                 # 降低温度, 减少胡说
        repetition_penalty=1.3,          # 提高重复惩罚
        no_repeat_ngram_size=4,          # 禁止 4-gram 重复
    )
    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    ).strip()
    elapsed = time.time() - st
    return response, elapsed


# ============ 跑对比 ============
results = []
for i, prompt in enumerate(PROMPTS, 1):
    print(f"\n{'=' * 60}")
    print(f"[{i}/{len(PROMPTS)}] {prompt}")
    print('-' * 60)
    
    a, ta = generate(model_A, tokenizer_mine, prompt)
    print(f"\n[A 自训] ({ta:.1f}s)")
    print(a)
    
    b, tb = generate(model_B, tokenizer_mine, prompt)
    print(f"\n[B LoRA] ({tb:.1f}s)")
    print(b)
    
    c, tc = generate(model_C, tokenizer_official, prompt)
    print(f"\n[C 官方] ({tc:.1f}s)")
    print(c)
    
    results.append((prompt, a, b, c))


# ============ 保存 Markdown ============
def md_escape(text):
    """让长文本能放进 Markdown 表格"""
    return text.replace('\n', '<br>').replace('|', '\\|')


with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
    f.write("# 三方对比测试: 自训 64M / 自训 64M+LoRA / 官方 minimind-3\n\n")
    
    f.write("## 实验配置\n\n")
    f.write("| 组别 | 模型 | 权重路径 | 数据规模 | 备注 |\n")
    f.write("|------|------|----------|----------|------|\n")
    f.write("| A 自训 | 自训 64M SFT | `./out/full_sft_768.pth` | 完整 mini 数据 | 基线模型 |\n")
    f.write("| B LoRA | A + 文档微调 | `./out/from_mine_768.pth` | 文档 QA 微调 | 测试 LoRA 增益 |\n")
    f.write("| C 官方 | minimind-3 | `gongjy/minimind-3` | 完整数据多 epoch | 强基座参照 |\n\n")
    
    f.write("**模型架构**: 三模型完全相同 (hidden_size=768, num_layers=8, ~64M 参数)\n\n")
    f.write("**生成参数**: temperature=0.6, top_p=0.85, repetition_penalty=1.3, no_repeat_ngram_size=4, max_new_tokens=400\n\n")
    
    f.write("---\n\n")
    
    f.write("## 第一部分: 通用能力对比 (8 题)\n\n")
    f.write("> 评估三个模型的基础对话、知识、推理、创作能力\n\n")
    f.write("| # | 问题 | A 自训 | B 自训+LoRA | C 官方 |\n")
    f.write("|---|------|--------|-------------|--------|\n")
    for i, (q, a, b, c) in enumerate(results[:DOC_START_IDX], 1):
        f.write(f"| {i} | {q} | {md_escape(a)} | {md_escape(b)} | {md_escape(c)} |\n")
    
    f.write("\n---\n\n")
    
    f.write("## 第二部分: 文档专有知识对比 (5 题)\n\n")
    f.write("> 评估 LoRA 是否成功学到了华科文档中的专有知识\n\n")
    f.write("| # | 问题 | A 自训 | B 自训+LoRA | C 官方 |\n")
    f.write("|---|------|--------|-------------|--------|\n")
    for i, (q, a, b, c) in enumerate(results[DOC_START_IDX:], 1):
        f.write(f"| {i} | {q} | {md_escape(a)} | {md_escape(b)} | {md_escape(c)} |\n")
    
    f.write("\n---\n\n")
    
    f.write("## 分析框架\n\n")
    f.write("### 通用问题分析\n\n")
    f.write("- **A vs C**: 验证 Scaling Law - 同架构下数据量对模型能力的决定性影响\n")
    f.write("- **A vs B**: 检查 LoRA 是否引发灾难性遗忘 (B 大幅退步说明遗忘严重)\n\n")
    
    f.write("### 文档问题分析\n\n")
    f.write("- **A vs B**: LoRA 的核心增益 - B 应能答出 A 完全不知道的文档知识\n")
    f.write("- **B vs C**: LoRA 的核心价值 - B 在文档专有问题上应优于训练数据更大的 C\n")
    f.write("- 即使 C 的通用能力强, 但它从未见过华科文档, 所以应答不出具体规则\n\n")

print(f"\n{'=' * 60}")
print(f"✅ 三方对比完成! 共测试 {len(PROMPTS)} 题")
print(f"📄 报告已保存到: {OUTPUT_MD}")
print(f"{'=' * 60}")