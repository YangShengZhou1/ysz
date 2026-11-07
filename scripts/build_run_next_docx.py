"""四项实验数据报告 docx。数据全部来自 outputs/experiments/2026-11-07_next。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from docx_helpers import bullet, caption, h1, h2, new_doc, para, subtitle, table, title

ROOT = HERE.parent
EXP = ROOT / "outputs/experiments/2026-11-07_next"
DOCS = ROOT.parents[1] / "docs/2026-11-07"
TARGET = DOCS / "23_四项实验数据报告_2026-11-07.docx"

S = json.loads((EXP / "summary.json").read_text(encoding="utf-8"))
P = json.loads((EXP / "partition_stats.json").read_text(encoding="utf-8"))
AC = {
    f"{tag}_r{r}": json.loads(
        (EXP / f"allcand_paper100-next-{tag}_r{r}.json").read_text(encoding="utf-8")
    )
    for tag in ("dir", "path") for r in (250, 500)
}


def kl(tag, r):
    return S["kl"][f"kl_paper100-next-{tag}_r{r}"]


def rp(tag, r):
    return S["replay"][f"replay_paper100-next-{tag}_r{r}"]


def delayed(csv_name):
    with (EXP / csv_name).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["target"], {})[r["branch"]] = r
    delayed_n = 0
    pairs = 0
    for b in by.values():
        n0 = float(b["none"]["ver_acc_before"])
        n1 = float(b["none"]["ver_acc_after"])
        for name in ("kl_top1", "select_best"):
            d0 = float(b[name]["ver_acc_before"]) - n0
            d1 = float(b[name]["ver_acc_after"]) - n1
            pairs += 1
            if d0 < -1e-12 and d1 > 1e-12:
                delayed_n += 1
    return f"{delayed_n}/{pairs}"


def split_sizes(csv_name):
    with (EXP / csv_name).open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["branch"] == "none"]
    return round(sum(int(r["select_n"]) for r in rows) / len(rows)), round(
        sum(int(r["verify_n"]) for r in rows) / len(rows)
    )


def build():
    doc = new_doc()
    title(doc, "四项实验数据报告")
    subtitle(doc, "2026-11-07 | CIFAR-10 | 100 客户端 | 500 轮 | 两格划分")

    h1(doc, "1 实验配置")
    table(doc, ["项", "Dirichlet 格", "Pathological 格"], [
        ["数据集", "CIFAR-10", "CIFAR-10"],
        ["客户端数", "100", "100"],
        ["数据划分", "my_part, alpha=0.3, 10 组组内同质", "n_cls, 每客户端 2 类, 10 组"],
        ["拓扑", "random, 每点 10 邻居", "random, 每点 10 邻居"],
        ["通信轮数", "500 (501 次迭代)", "500 (501 次迭代)"],
        ["本地 epoch", "5", "5"],
        ["batch size", "128", "128"],
        ["优化器", "SGD 0.1, lr_decay=1.0", "SGD 0.1, lr_decay=1.0"],
        ["共享范围", "只聚合 base, head 保留本地", "只聚合 base, head 保留本地"],
        ["原型", "初始化后固定, proto_update_each=0", "初始化后固定, proto_update_each=0"],
        ["快照轮次", "r1 / r250 / r500", "r1 / r250 / r500"],
        ["耗时", "14 h 36 m", "14 h 29 m"],
    ], widths=[3.2, 6.4, 6.4], size=8.5)
    caption(doc, "表 1 两格除数据划分外逐项相同。配置对照 ICME Experiment Setup。训练均 501/501 完成, 0 报错。")
    para(doc, "第 100 轮起由 change_neighbors 改边, 对应论文的动态拓扑方法。快照全部保留, 不再删除。", size=9.5)

    h1(doc, "2 数据划分与样本量")
    table(doc, ["项", "Dirichlet 格", "Pathological 格"], [
        ["每客户端样本量范围", "398 - 401", "399 - 401"],
        ["每客户端样本量均值", "399.8", "400.0"],
        ["每客户端非零类别数范围", "1 - 10", "1 - 7"],
        ["每客户端非零类别数均值", "5.7", "2.2"],
        ["划分生成方式", "10 个 Dirichlet(0.3) 先验, 组内 10 客户端共享", "每组随机 2 类, 先验硬置 1/2"],
    ], widths=[4.4, 5.8, 5.8], size=8.5)
    caption(doc, "表 2 每客户端样本量与各类别样本数。完整逐客户端明细见 partition_stats.json。")
    para(doc, "两类划分都由按类别先验分配样本, 样本量生成方式相同, 单人约 400 张。某类样本耗尽时会从其余类补齐, 因此 Pathological 格实际非零类别数会略高于 2。两条分支共用同一套分配代码, 可以互比。", size=9.5)

    h1(doc, "3 候选差异")
    table(doc, ["配置", "退化头", "零收益", "平均 Δacc (pp)", "目标内极差 (pp)", "regret (pp)"], [
        [f"{tag} r{r}",
         f"{kl(tag, r)['degenerate_frac']:.1%}",
         f"{kl(tag, r)['zero_frac']:.1%}",
         f"{kl(tag, r)['mean_delta_pp']:+.2f}",
         f"{kl(tag, r)['range_pp_mean']:.2f}",
         f"{kl(tag, r)['regret_pp_mean']:.2f}"]
        for tag in ("dir", "path") for r in (1, 250, 500)
    ], widths=[3.0, 2.4, 2.4, 3.0, 3.0, 2.2], size=8.5)
    caption(doc, "表 3 每快照 100 个目标, 每目标 20 个候选, 共 2000 对。主统计基于全部配对。")
    para(doc, "结果按目标客户端和快照轮次分别保留。本轮固定 seed 0 和只聚合 base, 不做种子与聚合范围变体, 所以没有这两列分组。", size=9.5)
    table(doc, ["配置", "KL 首选命中", "混算 Spearman", "目标内 Spearman"], [
        [f"{tag} r{r}", kl(tag, r)["top1_hit"],
         f"{kl(tag, r)['overall_rho']:+.3f}",
         f"{kl(tag, r)['target_rho_mean']:+.3f}"]
        for tag in ("dir", "path") for r in (1, 250, 500)
    ], widths=[4.0, 4.0, 4.0, 4.0], size=9)
    caption(doc, "表 4 排序指标。20 个候选下随机命中基线为 5/100。")

    h1(doc, "4 伙伴距离")
    d5, p5 = kl("dir", 500)["hop_mean_delta_pp"], kl("path", 500)["hop_mean_delta_pp"]
    dh = {h[0]: h for h in d5}
    ph = {h[0]: h for h in p5}
    table(doc, ["跳数", "Dirichlet 配对数", "Dirichlet 平均 Δacc (pp)", "Pathological 配对数", "Pathological 平均 Δacc (pp)"], [
        [str(h), dh.get(h, [0, 0, 0])[1], f"{dh.get(h, [0, 0, 0])[2]:+.2f}",
         ph.get(h, [0, 0, 0])[1], f"{ph.get(h, [0, 0, 0])[2]:+.2f}"]
        for h in sorted(set(list(dh) + list(ph)))
    ], widths=[2.0, 3.4, 4.0, 3.4, 3.2], size=8.5)
    caption(doc, "表 5 r500 按图最短跳数的平均聚合收益。")
    table(doc, ["配置", "r1", "r250", "r500"], [
        [tag, kl(tag, 1)["best_two_hop"], kl(tag, 250)["best_two_hop"], kl(tag, 500)["best_two_hop"]]
        for tag in ("dir", "path")
    ], widths=[4.0, 4.0, 4.0, 4.0], size=9)
    caption(doc, "表 6 每个目标事后最优伙伴落在两跳内的目标数, 分母 100。")

    h1(doc, "5 短程重放")
    rr = []
    for tag in ("dir", "path"):
        for r in (250, 500):
            s = rp(tag, r)
            b = s["branches"]
            g = s["gain_vs_none_after"]
            rr.append([
                f"{tag} r{r}",
                f"{b['none']['ver_acc_before']:.3f} / {b['none']['ver_acc_after']:.3f}",
                f"{b['kl_top1']['ver_acc_before']:.3f} / {b['kl_top1']['ver_acc_after']:.3f}",
                f"{b['select_best']['ver_acc_before']:.3f} / {b['select_best']['ver_acc_after']:.3f}",
                f"{g['kl_top1']['mean_pp']:+.2f}",
                f"{g['select_best']['mean_pp']:+.2f}",
            ])
    table(doc, ["配置", "不聚合 即时/1 epoch 后", "KL 伙伴 即时/后", "选择集最优 即时/后", "KL 增益 (pp)", "选择增益 (pp)"],
          rr, widths=[2.4, 3.4, 3.4, 3.4, 1.9, 1.9], size=8)
    caption(doc, "表 7 核查集准确率。数值为聚合后即时 / 1 个本地 epoch 后。增益是重放后相对不聚合分支的差。")
    para(doc, "所有分支从同一模型出发, 只聚合 base, 保留目标自己的 head, 样本顺序固定, 优化器每次重建, 评价统一用 eval 模式。100 个目标全部保留, 不做二次筛选。", size=9.5)
    para(doc, "分支起点核对: 模型用 GroupNorm 替换 BatchNorm, state_dict 中没有 BN 运行统计键。对 dir r500 的前 5 个目标核对, head 的 2 个键在各分支起点逐元素相同, 60 个 base 可学习参数按聚合权重改变, 没有计划外的差异。核对输出见 bn_check.json。", size=9.5)

    h1(doc, "6 挑验分离")
    sr = []
    for tag in ("dir", "path"):
        for r in (250, 500):
            name = f"replay_paper100-next-{tag}_r{r}"
            s = rp(tag, r)
            sel, ver = split_sizes(name + ".csv")
            g = s["gain_vs_none_after"]["select_best"]
            sr.append([
                f"{tag} r{r}", f"{sel} / {ver}",
                f"{s['select_best_is_kl_top1_frac']:.0%}",
                f"{g['improved']} / {g['degraded']} / {g['flat']}",
                delayed(name + ".csv"),
            ])
    table(doc, ["配置", "选择集 / 核查集", "选择与 KL 重合", "选择集增益 改善/退化/不明显", "即时落后训练后改善"],
          sr, widths=[2.6, 3.2, 3.0, 4.4, 3.2], size=8.5)
    caption(doc, "表 8 留出验证集按类别分层固定切两半。伙伴择优只看选择集, 成绩只报核查集, 测试集未用于选伙伴。")
    para(doc, "选择依据是聚合后选择集交叉熵损失最小的伙伴, 选定后固定, 不根据后续效果重挑。每个候选的即时选择集损失保存在逐目标 CSV 的 candidates_json 字段。三个固定分支用于最终对照, 全部 4 个候选的训练后输出另行采集, 见 6.1。", size=9.5)

    h2(doc, "6.1 全部候选的重放输出")
    table(doc, ["配置", "即时平均增益 (pp)", "1 epoch 后平均增益 (pp)", "即时落后训练后改善", "即时占优训练后变差", "有延迟候选的目标数", "事后最优的 KL 平均排名"], [
        [f"{tag} r{r}",
         f"{AC[f'{tag}_r{r}']['mean_gain_before_pp']:+.2f}",
         f"{AC[f'{tag}_r{r}']['mean_gain_after_pp']:+.2f}",
         AC[f'{tag}_r{r}']['delayed_gain'],
         AC[f'{tag}_r{r}']['reversed_gain'],
         AC[f'{tag}_r{r}']['targets_with_delayed_candidate'],
         f"{AC[f'{tag}_r{r}']['mean_kl_rank_of_best_after']:.2f}"]
        for tag in ("dir", "path") for r in (250, 500)
    ], widths=[2.0, 2.4, 2.8, 2.4, 2.4, 1.8, 2.2], size=8)
    caption(doc, "表 9 每目标 4 个候选全部重放, 每格 100 目标 400 对。增益相对同目标的不聚合分支。KL 平均排名在 4 个候选中取值 1 到 4, 随机期望为 2.5。")

    h1(doc, "7 口径")
    for text in [
        "KL 样本: 每快照 2000 对, 来自 r1 / r250 / r500 三个快照, 每目标按固定种子从其余 99 个客户端抽 20 个候选。收益是只聚合 base 后, 在目标客户端 20% 留出验证集上的 Δacc。主统计用全部 2000 行, 退化比例单独列出。",
        "轮数: 500 轮对应 501 次迭代。三项快照分别记为 r1 / r250 / r500。",
        "通信: 稠密张量按元素数乘每元素字节数再乘交换次数计。base 单份 11,168,832 元素, fp32 下 42.6 MiB; 原型 512 维, 2,048 B; head 5,130 元素不共享。r1 快照 526 条图边, 958 个选中邻居槽位, 每轮上界约 42.8 GB, 500 轮上界约 21.4 TB。控制消息, 多跳转发, 拓扑维护未计入。",
        "原型: 按 11-05 指导保持初始化后固定。三个快照的 2000 对 KL 逐对相同。",
        "快照: 六个 round 目录在服务器 /home/seczone/ysz/outputs/checkpoints 下, 各 4.2G, 共 25.2G。模型文件不打包上传, 元数据与索引已随交付材料提供。",
    ]:
        bullet(doc, text, size=9)

    h1(doc, "8 分析与推测")
    for text in [
        "两格的模型都练成熟了。r1 退化头 84% 到 92%, 零收益 91% 到 94%; 到 r250 / r500 退化头降到 1% 以下, 收益分布打开。r1 不适合做排序或重放判断。",
        "同一目标换伙伴的收益差别很大。r500 目标内极差 Dirichlet 格 6.34 pp, Pathological 格 17.76 pp, r250 的 Pathological 格达到 27.58 pp。差别量级远大于抽样噪声。",
        "KL 的排序能力没有随训练恢复。Dirichlet 格 r500 首选命中 5/100, 等于随机基线; 目标内 Spearman +0.067。Pathological 格命中升到 22/100, 但目标内 Spearman 为 -0.023。KL 能区分标签分布是否同源, 排不出同类候选的次序。",
        "差别经过 1 个本地 epoch 仍然存在, 但挑出来的优势没有带到核查集。Dirichlet 格 r500 的 KL 伙伴和选择集最优伙伴重放后分别比不聚合低 6.25 pp 和 7.15 pp。Pathological 格两轮的表现接近不聚合。选择集挑出的伙伴与 KL 挑的只有 28% 到 47% 重合, 两个信号都不稳定。",
        "跳数在两格都比 KL 更能看出平均收益。r500 的 Dirichlet 格从 1 跳 +2.17 pp 降到 3 跳 +1.18 pp; Pathological 格从 1 跳 +0.16 pp 降到 4 跳 -4.31 pp。事后最优伙伴约 7 成落在两跳内, 远跳平均有害。按 11-05 指导的分支判断, 多跳发现的优先级可以降低。",
        "推测瓶颈在选伙伴的信号, 不在候选池大小。候选本身有可用差异, 但 KL 和少量选择集反馈都不足以在核查集上稳定选对。后续若要设计采纳规则, 需要比单批选择集损失更稳的验证信号, 并把取得候选模型的通信成本一并计入。",
        "延迟收益线索确实存在。每格 400 对里有 36 到 60 对即时不占优、训练 1 个 epoch 后转为改善, 涉及 26 到 36 个目标。但反向变化更多, dir r500 有 92 对即时占优训练后变差。平均增益从即时 +1.78 pp 转为重放后 -6.03 pp。事后最优伙伴的 KL 平均排名为 2.00 到 2.24, 接近 4 个候选的随机期望 2.5。",
    ]:
        bullet(doc, text, size=9.5)

    h2(doc, "8.1 当前阻塞与待定事项")
    for text in [
        "老师的三条分岔逐条对照。收益差别在核查集和 1 个 epoch 后确实留得住, 但用 KL 或选择集 loss 挑出来的优势留不住, 第一条不能判为成立。",
        "即时看到的聚合收益在重放后普遍反转。dir r500 即时平均 +1.52 pp, 重放后 KL 伙伴相对不聚合为 -6.25 pp。按老师的第二条分岔, 应降低精细选邻的优先级, 转去分析聚合与本地训练的关系。",
        "有帮助的伙伴不主要在远处。两格约 7 成的事后最优伙伴落在两跳内, 跳数越远平均收益越低, 不需要重点发展多跳。",
        "当前阻塞: 按结果落在第二条分岔, 但下一项机制按老师要求由双方一起确定, 本轮不自行设计采纳规则。若要继续走用反馈选伙伴这条路, 需要先解决选择集信号泛化不到核查集的问题, 并把取得候选模型的下载通信成本计入。",
    ]:
        bullet(doc, text, size=9.5)

    h1(doc, "9 文件与代码")
    table(doc, ["内容", "位置"], [
        ["本报告与总体数据报告", "docs/2026-11-07/23_..., docs/2026-11-06/22_..."],
        ["逐配对原始输出", "outputs/2026-11-07_next/kl_*.csv, kl_*.json"],
        ["重放逐目标输出", "outputs/2026-11-07_next/replay_*.csv, replay_*.json"],
        ["汇总与划分统计", "outputs/2026-11-07_next/summary.json, partition_stats.json"],
        ["快照元数据", "outputs/2026-11-07_next/metadata/*_metadata.json"],
        ["全候选重放输出", "outputs/2026-11-07_next/allcand_*.csv, allcand_*.json"],
        ["分支起点核对", "outputs/2026-11-07_next/bn_check.json"],
        ["交付压缩包", "deliverables/反馈_2026-11-07.zip"],
        ["训练日志", "outputs/2026-11-07_next/logs/"],
        ["代码", "scripts/kl_benefit_distribution.py, same_target_replay_paper100.py, paper100_candidate_diff.py, _remote_summarize_next.py"],
        ["仓库", "https://github.com/YangShengZhou1/ysz"],
    ], widths=[4.6, 11.4], size=8.5)
    caption(doc, "表 10 交付材料位置。模型权重不上传 GitHub, 位置见第 7 节。")

    h1(doc, "10 旧方案材料与配置来源")

    h2(doc, "10.1 材料清单")
    table(doc, ["材料", "提供的内容"], [
        ["ICME 2025 论文 (copy.tex)", "方法框架与全部实验设置, 三种划分, 四种拓扑, 100 客户端, 500 轮, 骨干与超参"],
        ["毕业论文终稿", "第三章完整方法描述, 消融数字, 拓扑示意图"],
        ["扩刊初稿 (tmm_new.tex)", "三种划分与四种拓扑的扩展描述, 500 轮收敛表"],
        ["答辩 PPT", "方法示意与主要结果图"],
        ["指导记录 (10-15, 11-04, 11-05)", "研究前提, 本轮四项比较的设计, 原型固定的口径"],
        ["资料审阅与问题清单 (docs/06)", "论文设定与代码实现差异的逐条核对结果"],
    ], widths=[5.2, 10.8], size=8.5)
    caption(doc, "表 11 旧方案材料清单及各自提供的内容。")
    para(doc, "扩刊初稿的数字未经验证, 只用来理解设定, 不作为结果依据。", size=9.5)

    h2(doc, "10.2 配置来源")
    table(doc, ["项", "取值", "来源"], [
        ["客户端 / 分组 / 轮数", "100 / 10 组 / 500", "ICME Experiment Setup"],
        ["本地 epoch / batch / 优化器", "5 / 128 / SGD 0.1", "ICME Implementation Details"],
        ["骨干与共享范围", "ResNet-18, 只共享特征提取器", "ICME Implementation Details"],
        ["初始图", "随机连通图, 每点 10 邻居", "ICME Graph Topology and Communication Setup, 代码 cs=random"],
        ["动态改边", "第 100 轮起", "论文 Performance with Dynamic Topology, 代码 change_neighbors"],
        ["原型", "初始化后固定", "11-05 指导第 63 行, 代码 proto_update_each=0"],
        ["学习率衰减", "关闭", "论文只写 lr 0.1, 代码默认每轮乘 0.998"],
        ["快照轮次", "r1 / r250 / r500", "本轮自定义, 论文未报快照轮次"],
        ["20 候选, 验证集二分, 1 epoch 重放", "本轮四项比较", "11-05 指导第三节"],
    ], widths=[3.6, 4.0, 8.4], size=8.5)
    caption(doc, "表 12 配置逐项来源。未在论文中出现的实现细节单独列出。")

    h2(doc, "10.3 为什么有 Dirichlet 和 Pathological 两格")
    para(doc, "论文在三个数据集下都按两种数据划分报结果, 主表的列就是 Dirichlet 和 Pathological。它们不是两个方法, 是同一方法在两种异质强度下的对照。", size=9.5)
    para(doc, "Dirichlet 格取 alpha=0.3。每个客户端按抽出的类别比例拿样本, 主类占比高, 其他类仍有一些, 属于软偏斜。代码用 my_part 实现, 先抽 10 个 Dirichlet 先验, 组内 10 个客户端共用一个先验, 对应论文的 10 组组内同质。实测每客户端约 400 张样本, 非零类别均值 5.7。", size=9.5)
    para(doc, "Pathological 格对 CIFAR-10 限定每客户端 2 类, CIFAR-100 是 20 类, Tiny-ImageNet 是 40 类。类别数是硬上限, 异质性更强。代码用 n_cls 加 partition_alpha=2 实现。实测每客户端约 400 张样本, 非零类别均值 2.2; 某类样本耗尽时会从其余类补齐, 所以个别客户端会略高于 2 类。", size=9.5)
    para(doc, "选这两格有三个理由。论文主表就是这两列, 这是复现的最基本单位; 只换划分方式, 其余十项不动, 可以直接看出结论是否依赖划分; 两档异质强度差距大, 能检验 KL 的粗筛能力在更强异质性下是否变化。实测 KL 首选命中从 Dirichlet 格的 5/100 升到 Pathological 格的 22/100, 说明这个对照提供了新信息。", size=9.5)
    para(doc, "第三种 Cross-Dataset 划分混合 CIFAR-10 和 CIFAR-100, 涉及两套标签空间, 本轮不做。早期脚本用的 dir 是全局 Dirichlet, 不构造 10 组, 与论文口径不符, 本轮已不用。", size=9.5)

    h2(doc, "10.4 旧方案留下的空白")
    for text in [
        "论文里的协作价值由原型差异定义, 证据是整套框架对比和消融, 没有验证集选伙伴, 同一目标换伙伴这类实验。本轮四项比较补的就是排序层的证据。",
        "毕业论文消融写总通信 115.5 MB, 但 ResNet-18 单份 base 就 42.6 MiB, 100 客户端, 10 邻居, 500 轮的模型交换量级远大于此。两者统计口径不同, 不能直接比较。本轮按实际张量元素数重新计量, 并把未计入项写明。",
        "只共享 base, 保留本地分类头在旧方案和相关工作中已有, 不作为新贡献。",
        "my_part 是 10 个共享先验近似分组, 不等同于严格的组内同质。报告按实际分布描述, 不写成完全复现分组。",
    ]:
        bullet(doc, text, size=9.5)

    h2(doc, "10.5 取舍")
    para(doc, "按 11-05 指导和检查清单, 只跑论文里出现的格子。alpha 取 0.1 或 0.5, seed 变体, 客户端数变体都不是论文设定, 不做。Ring, Grid, Exp 三种拓扑以及 CIFAR-100, Tiny-ImageNet 留到后续。", size=9.5)

    DOCS.mkdir(parents=True, exist_ok=True)
    doc.save(TARGET)
    print("written:", TARGET)


if __name__ == "__main__":
    build()
