"""将 mydoc.docx 转成 LoRA 训练数据（SFT 格式）"""
import json
from docx import Document

INPUT_DOCX = 'mydoc.docx'
OUTPUT_JSONL = 'exp_lora/data/my_sft.jsonl'
CHUNK_CHAR_LEN = 400

print(f"读取 {INPUT_DOCX} ...")
doc = Document(INPUT_DOCX)
paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
full_text = '\n'.join(paragraphs)
print(f"全文字数：{len(full_text)}")
print(f"段落数：{len(paragraphs)}")

chunks = []
buf = ''
for para in paragraphs:
    if len(buf) + len(para) + 1 < CHUNK_CHAR_LEN:
        buf += para + '\n'
    else:
        if buf.strip():
            chunks.append(buf.strip())
        buf = para + '\n'
if buf.strip():
    chunks.append(buf.strip())

print(f"切分成 {len(chunks)} 段")

INSTRUCTIONS = [
    "请简述以下内容相关的信息。",
    "请根据你所知道的内容详细介绍。",
    "请说明相关内容的要点。",
    "请展开讲讲这方面内容。",
    "请介绍相关的知识。",
]

with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for i, c in enumerate(chunks):
        instr = INSTRUCTIONS[i % len(INSTRUCTIONS)]
        sample = {
            "conversations": [
                {"role": "user", "content": instr},
                {"role": "assistant", "content": c}
            ]
        }
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"✅ 已保存 {len(chunks)} 条训练样本到 {OUTPUT_JSONL}")