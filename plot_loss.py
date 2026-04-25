import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_one(csv_path, title, output):
    if not os.path.exists(csv_path):
        print(f"找不到 {csv_path}，跳过")
        return
    df = pd.read_csv(csv_path, names=['epoch', 'step', 'loss', 'lr'])
    window = min(50, max(1, len(df) // 20))
    df['loss_smooth'] = df['loss'].rolling(window=window, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df.index, df['loss'], alpha=0.3, color='steelblue', label='原始 loss')
    ax1.plot(df.index, df['loss_smooth'], 'r-', linewidth=2, label=f'平滑 loss (window={window})')
    ax1.set_xlabel('训练步数 (step)')
    ax1.set_ylabel('Loss', color='steelblue')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df['lr'], 'g--', alpha=0.6, label='学习率')
    ax2.set_ylabel('Learning Rate', color='green')
    ax2.legend(loc='center right')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存 {output}")
    print(f"  数据点数: {len(df)}")
    print(f"  起始 loss: {df['loss'].iloc[0]:.4f}")
    print(f"  最终 loss: {df['loss'].iloc[-1]:.4f}")
    print(f"  平滑最终 loss: {df['loss_smooth'].iloc[-1]:.4f}")

plot_one('./out/pretrain_loss.csv', 'MiniMind2-Small 预训练 Loss 曲线', 'pretrain_loss.png')
plot_one('./out/sft_loss.csv', 'MiniMind2-Small SFT Loss 曲线', 'sft_loss.png')