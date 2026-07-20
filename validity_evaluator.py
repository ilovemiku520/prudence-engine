# ================================================================
# validity_evaluator.py
# Validity 效果评估框架 —— 适当性、意图、融合决策三维评估
# 版本: v1.0
# ================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json
from enum import Enum

# 可视化
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False

# 机器学习指标
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve

from loguru import logger


# ================================================================
# 1. 数据模型定义
# ================================================================

@dataclass
class SuitabilityEvaluationSample:
    """适当性评估样本"""
    customer_id: str
    product_id: str
    customer_risk: str          # C1-C5
    product_risk: str           # R1-R5
    matrix_result: str          # 矩阵原始判定: ALLOW/RESTRICTED/FORBID
    final_decision: str         # 实际系统输出: ALLOW/RESTRICTED/FORBID
    is_manual_review: bool      # 是否经过人工复核
    manual_verdict: Optional[str] = None  # 人工复核结论: CORRECT/INCORRECT
    # 用于误拦截分析
    should_allow: Optional[bool] = None   # 业务上是否应允许（由专家标注）

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class IntentEvaluationSample:
    """意图评估样本（离线/在线通用）"""
    customer_id: str
    feature_time: datetime
    intent_score: float          # 模型输出的意图分
    converted: int               # 在观察窗口内是否购买（0/1）
    # 在线特有字段
    is_recommended_by_model: bool = False  # 是否被模型推荐（仅在线A/B测试）
    is_accepted_by_adviser: bool = False   # 理财经理是否采纳推荐

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class FusionEvaluationSample:
    """融合决策评估样本"""
    customer_id: str
    product_id: str
    action: str                  # PROACTIVE_CLOSING / NURTURE_CONTENT / LOW_PRIORITY / HUMAN_REVIEW / BLOCK
    intent_score: float
    suitability_level: str
    converted: int               # 是否最终成交（0/1）
    complaint_received: int      # 是否收到投诉（0/1）
    compliance_review_passed: Optional[int] = None  # 合规事后复核是否通过（仅对拦截/受限样本）

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ================================================================
# 2. 评估器主类
# ================================================================

class ValidityEvaluator:
    """
    鉴衡评估框架
    支持：
      - 适当性子系统指标：违规拦截准确率、误拦截率
      - 意图子系统指标：AUC、KS、分层转化率单调性、Lift曲线
      - 融合决策指标：促单转化率、投诉率、合规复核通过率
    """

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"ValidityEvaluator 初始化，报告输出目录: {output_dir}")

    # ================================================================
    # 2.1 适当性评估
    # ================================================================
    def evaluate_suitability(self, samples: List[SuitabilityEvaluationSample]) -> Dict:
        """
        评估适当性子系统
        核心指标：
          - 违规拦截准确率 = 系统拦截且人工判定应该拦截 / 总拦截数
          - 误拦截率 = 系统拦截但人工判定应该允许 / 总样本数（或总拦截数）
        """
        df = pd.DataFrame([s.to_dict() for s in samples])

        # 如果有手工标注的 should_allow，则计算
        if 'should_allow' in df.columns and df['should_allow'].notna().any():
            # 系统禁止但实际应允许 => 误拦截
            false_block = df[(df['final_decision'] == 'FORBID') & (df['should_allow'] == True)]
            # 系统禁止且实际应禁止 => 正确拦截
            true_block = df[(df['final_decision'] == 'FORBID') & (df['should_allow'] == False)]

            total_block = len(df[df['final_decision'] == 'FORBID'])
            if total_block > 0:
                block_accuracy = len(true_block) / total_block
                false_block_rate = len(false_block) / len(df)  # 误拦截占总样本比例
            else:
                block_accuracy = 1.0
                false_block_rate = 0.0
        else:
            # 若无标注，用人工复核 verdict 代替（假设 manual_verdict 有值）
            if 'manual_verdict' in df.columns and df['manual_verdict'].notna().any():
                df_correct = df[df['manual_verdict'] == 'CORRECT']
                df_incorrect = df[df['manual_verdict'] == 'INCORRECT']
                total_block = len(df[df['final_decision'] == 'FORBID'])
                if total_block > 0:
                    true_block = df[(df['final_decision'] == 'FORBID') & (df['manual_verdict'] == 'CORRECT')]
                    block_accuracy = len(true_block) / total_block
                    false_block_rate = len(df[(df['final_decision'] == 'FORBID') & (df['manual_verdict'] == 'INCORRECT')]) / len(df)
                else:
                    block_accuracy = 1.0
                    false_block_rate = 0.0
            else:
                logger.warning("无人工标注，无法计算准确率与误拦截率")
                block_accuracy = None
                false_block_rate = None

        # 受限（RESTRICTED）转人工的复核通过率（如果有限制样本）
        restricted_samples = df[df['final_decision'] == 'RESTRICTED']
        if len(restricted_samples) > 0 and 'manual_verdict' in df.columns:
            restricted_correct = restricted_samples[restricted_samples['manual_verdict'] == 'CORRECT']
            restricted_pass_rate = len(restricted_correct) / len(restricted_samples)
        else:
            restricted_pass_rate = None

        result = {
            "block_accuracy": block_accuracy,      # 违规拦截准确率
            "false_block_rate": false_block_rate,  # 误拦截率
            "restricted_pass_rate": restricted_pass_rate,
            "total_samples": len(df),
            "block_count": len(df[df['final_decision'] == 'FORBID']),
            "restricted_count": len(df[df['final_decision'] == 'RESTRICTED']),
            "allow_count": len(df[df['final_decision'] == 'ALLOW'])
        }
        return result

    # ================================================================
    # 2.2 意图评估（离线指标）
    # ================================================================
    @staticmethod
    def evaluate_intent_offline(samples: List[IntentEvaluationSample]) -> Dict:
        """
        离线意图评估：AUC, KS, 分层单调性
        """
        df = pd.DataFrame([s.to_dict() for s in samples])
        if len(df) == 0:
            return {"error": "无样本"}

        score = df['intent_score']
        label = df['converted']

        # AUC
        try:
            auc = roc_auc_score(label, score)
        except:
            auc = 0.5

        # KS = max(|TPR - FPR|)
        fpr, tpr, thresholds = roc_curve(label, score)
        ks = max(tpr - fpr)

        # 分层转化率单调性
        monotonic, conversion_by_decile = ValidityEvaluator.check_monotonicity(df, score_col='intent_score', label_col='converted')

        # 分层转化率表格
        decile_table = conversion_by_decile.to_dict()

        return {
            "auc": auc,
            "ks": ks,
            "monotonic_increasing": monotonic,
            "decile_conversion": decile_table,
            "sample_count": len(df),
            "positive_rate": label.mean()
        }

    @staticmethod
    def check_monotonicity(df: pd.DataFrame, score_col: str, label_col: str, n_bins: int = 10) -> Tuple[bool, pd.Series]:
        """
        检查意图分数分层的转化率是否单调递增
        返回：(是否单调, 各分层的转化率)
        """
        # 确保分数在[0,1]区间，若超出则截断
        scores = df[score_col].clip(0, 1)
        # 等频分箱
        try:
            df['decile'] = pd.qcut(scores, q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            # 若重复值过多，改用等宽分箱
            df['decile'] = pd.cut(scores, bins=n_bins, labels=False)

        # 计算每层转化率
        conversion = df.groupby('decile')[label_col].mean()
        # 补全缺失的分层（若有）
        full_index = range(n_bins)
        conversion = conversion.reindex(full_index, fill_value=0.0)

        # 检查是否单调递增（允许相等）
        is_monotonic = all(conversion.iloc[i] <= conversion.iloc[i+1] for i in range(len(conversion)-1))
        return is_monotonic, conversion

    # ================================================================
    # 2.3 意图评估（在线指标）
    # ================================================================
    @staticmethod
    def evaluate_intent_online(samples: List[IntentEvaluationSample], random_samples: List[IntentEvaluationSample] = None) -> Dict:
        """
        在线意图评估：分层转化率、Lift曲线、理财经理采纳率
        需要对比实验组（有意图分推荐）和对照组（无差别推荐）
        """
        df = pd.DataFrame([s.to_dict() for s in samples])
        if len(df) == 0:
            return {"error": "无实验组样本"}

        # 分层转化率（同离线）
        monotonic, conversion_by_decile = ValidityEvaluator.check_monotonicity(df, 'intent_score', 'converted')

        # Lift曲线：与基准转化率（对照组）对比
        if random_samples is not None:
            df_control = pd.DataFrame([s.to_dict() for s in random_samples])
            base_conversion = df_control['converted'].mean()
            # 计算每层的lift
            lift = conversion_by_decile / base_conversion if base_conversion > 0 else np.nan
            lift_curve = lift.to_dict()
        else:
            # 若无对照组，用整体平均作为基准
            base_conversion = df['converted'].mean()
            lift = conversion_by_decile / base_conversion if base_conversion > 0 else np.nan
            lift_curve = lift.to_dict()

        # 理财经理采纳率：推荐被采纳的比例
        if 'is_accepted_by_adviser' in df.columns:
            adviser_acceptance = df['is_accepted_by_adviser'].mean()
        else:
            adviser_acceptance = None

        return {
            "monotonic_increasing": monotonic,
            "decile_conversion": conversion_by_decile.to_dict(),
            "lift_curve": lift_curve,
            "base_conversion_rate": base_conversion,
            "adviser_acceptance": adviser_acceptance,
            "sample_count": len(df)
        }

    # ================================================================
    # 2.4 融合决策整体评估
    # ================================================================
    @staticmethod
    def evaluate_fusion(samples: List[FusionEvaluationSample]) -> Dict:
        """
        融合决策整体指标：
          - 各动作（PROACTIVE_CLOSING等）的转化率
          - 投诉率（按动作）
          - 合规拦截事后复核通过率（针对BLOCK和HUMAN_REVIEW）
        """
        df = pd.DataFrame([s.to_dict() for s in samples])

        # 按动作计算转化率
        action_conversion = df.groupby('action')['converted'].mean().to_dict()
        # 按动作计算投诉率
        action_complaint = df.groupby('action')['complaint_received'].mean().to_dict()

        # 合规复核通过率（仅对需要复核的样本）
        compliance_pass = None
        if 'compliance_review_passed' in df.columns:
            review_samples = df[df['compliance_review_passed'].notna()]
            if len(review_samples) > 0:
                compliance_pass = review_samples['compliance_review_passed'].mean()

        # 整体转化率
        overall_conversion = df['converted'].mean()
        overall_complaint = df['complaint_received'].mean()

        result = {
            "action_conversion_rate": action_conversion,
            "action_complaint_rate": action_complaint,
            "compliance_review_pass_rate": compliance_pass,
            "overall_conversion_rate": overall_conversion,
            "overall_complaint_rate": overall_complaint,
            "total_samples": len(df)
        }
        return result

    # ================================================================
    # 2.5 综合报告生成
    # ================================================================
    def generate_full_report(
        self,
        suitability_samples: List[SuitabilityEvaluationSample],
        intent_offline_samples: List[IntentEvaluationSample],
        intent_online_samples: List[IntentEvaluationSample],
        fusion_samples: List[FusionEvaluationSample],
        random_control_samples: Optional[List[IntentEvaluationSample]] = None,
        report_name: str = "validity_report"
    ) -> Dict:
        """
        生成完整评估报告（含各子系统指标汇总）
        """
        logger.info("生成综合评估报告...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "suitability": self.evaluate_suitability(suitability_samples),
            "intent_offline": self.evaluate_intent_offline(intent_offline_samples),
            "intent_online": self.evaluate_intent_online(intent_online_samples, random_control_samples),
            "fusion": self.evaluate_fusion(fusion_samples)
        }

        # 打印摘要
        self._print_report_summary(report)

        # 保存为JSON
        import json
        report_path = f"{self.output_dir}/{report_name}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"报告已保存至: {report_path}")

        # 生成Excel报表（若pandas可用）
        self._export_excel(report, report_name)

        # 可视化（若matplotlib可用）
        if VISUAL_AVAILABLE:
            self._plot_intent_monotonic(intent_offline_samples, report_name)
            self._plot_lift_curve(intent_online_samples, random_control_samples, report_name)

        return report

    # ================================================================
    # 2.6 辅助：报告打印与导出
    # ================================================================
    def _print_report_summary(self, report: Dict):
        print("\n" + "=" * 60)
        print(" Validity 评估报告摘要")
        print("=" * 60)

        # 适当性
        s = report.get('suitability', {})
        print("\n【适当性子系统】")
        print(f"  违规拦截准确率: {s.get('block_accuracy', 'N/A')}")
        print(f"  误拦截率: {s.get('false_block_rate', 'N/A')}")
        print(f"  受限转人工通过率: {s.get('restricted_pass_rate', 'N/A')}")

        # 意图离线
        io = report.get('intent_offline', {})
        print("\n【意图子系统（离线）】")
        print(f"  AUC: {io.get('auc', 'N/A'):.4f}")
        print(f"  KS: {io.get('ks', 'N/A'):.4f}")
        print(f"  分层单调性: {io.get('monotonic_increasing', 'N/A')}")
        if 'decile_conversion' in io:
            conv = io['decile_conversion']
            print(f"  分层转化率 (前3层 vs 后3层): {conv.get(0, 0):.3f} -> {conv.get(9, 0):.3f}")

        # 意图在线
        ion = report.get('intent_online', {})
        print("\n【意图子系统（在线）】")
        print(f"  理财经理采纳率: {ion.get('adviser_acceptance', 'N/A')}")
        if 'lift_curve' in ion:
            lift = ion['lift_curve']
            if lift:
                print(f"  Lift曲线 (最高层lift): {max(lift.values()):.2f}")

        # 融合决策
        f = report.get('fusion', {})
        print("\n【融合决策整体】")
        print(f"  整体转化率: {f.get('overall_conversion_rate', 0):.3f}")
        print(f"  整体投诉率: {f.get('overall_complaint_rate', 0):.3f}")
        print(f"  合规复核通过率: {f.get('compliance_review_pass_rate', 'N/A')}")
        if 'action_conversion_rate' in f:
            print("  各动作转化率:")
            for act, rate in f['action_conversion_rate'].items():
                print(f"    {act}: {rate:.3f}")
        print("=" * 60)

    def _export_excel(self, report: Dict, report_name: str):
        """将报告导出为Excel"""
        try:
            with pd.ExcelWriter(f"{self.output_dir}/{report_name}.xlsx", engine='openpyxl') as writer:
                # 适当性
                pd.DataFrame([report['suitability']]).to_excel(writer, sheet_name='适当性', index=False)
                # 意图离线
                pd.DataFrame([report['intent_offline']]).to_excel(writer, sheet_name='意图离线', index=False)
                # 意图在线
                pd.DataFrame([report['intent_online']]).to_excel(writer, sheet_name='意图在线', index=False)
                # 融合
                pd.DataFrame([report['fusion']]).to_excel(writer, sheet_name='融合决策', index=False)
            logger.info(f"Excel报告已导出: {self.output_dir}/{report_name}.xlsx")
        except Exception as e:
            logger.warning(f"Excel导出失败: {e}")

    def _plot_intent_monotonic(self, samples: List[IntentEvaluationSample], report_name: str):
        """绘制分层转化率单调性图"""
        if not samples:
            return
        df = pd.DataFrame([s.to_dict() for s in samples])
        _, conv = self.check_monotonicity(df, 'intent_score', 'converted')
        plt.figure(figsize=(8, 5))
        plt.plot(conv.index, conv.values, marker='o', linestyle='-', color='steelblue')
        plt.title('意图分层转化率单调性检验')
        plt.xlabel('分数分位层 (0=最低, 9=最高)')
        plt.ylabel('转化率')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{report_name}_monotonic.png")
        plt.close()
        logger.info(f"单调性图已保存: {self.output_dir}/{report_name}_monotonic.png")

    def _plot_lift_curve(self, samples: List[IntentEvaluationSample], control_samples: List[IntentEvaluationSample], report_name: str):
        """绘制Lift曲线"""
        if not samples:
            return
        df = pd.DataFrame([s.to_dict() for s in samples])
        _, conv = self.check_monotonicity(df, 'intent_score', 'converted')
        if control_samples:
            df_ctrl = pd.DataFrame([s.to_dict() for s in control_samples])
            base = df_ctrl['converted'].mean()
        else:
            base = df['converted'].mean()

        lift = conv / base if base > 0 else np.nan
        plt.figure(figsize=(8, 5))
        plt.plot(conv.index, lift, marker='s', linestyle='-', color='darkorange')
        plt.axhline(y=1.0, color='gray', linestyle='--', label='Baseline (1.0)')
        plt.title('Lift曲线 (相对于基准转化率)')
        plt.xlabel('分数分位层')
        plt.ylabel('Lift')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{report_name}_lift.png")
        plt.close()
        logger.info(f"Lift曲线已保存: {self.output_dir}/{report_name}_lift.png")


# ================================================================
# 3. 模拟数据生成器（用于演示）
# ================================================================

class EvaluationDataGenerator:
    """生成模拟评估数据"""

    @staticmethod
    def generate_suitability_samples(n: int = 1000) -> List[SuitabilityEvaluationSample]:
        samples = []
        risk_levels = ['C1','C2','C3','C4','C5']
        product_risks = ['R1','R2','R3','R4','R5']
        for i in range(n):
            c_risk = np.random.choice(risk_levels)
            p_risk = np.random.choice(product_risks)
            # 模拟矩阵判定（简单的：风险等级必须<=客户等级）
            c_idx = risk_levels.index(c_risk)
            p_idx = product_risks.index(p_risk)
            if p_idx <= c_idx:
                matrix_result = 'ALLOW'
            elif p_idx == c_idx + 1:
                matrix_result = 'RESTRICTED'
            else:
                matrix_result = 'FORBID'
            # 模拟系统决策（可能误判）
            if np.random.rand() < 0.02:  # 2%误判率
                if matrix_result == 'ALLOW':
                    final_decision = 'FORBID'
                elif matrix_result == 'FORBID':
                    final_decision = 'ALLOW'
                else:
                    final_decision = matrix_result
            else:
                final_decision = matrix_result

            # 人工标注是否应该允许（以矩阵为基准）
            should_allow = (p_idx <= c_idx)

            # 人工复核（假设部分样本复核）
            manual_verdict = None
            if np.random.rand() < 0.3:  # 30%复核
                manual_verdict = 'CORRECT' if (final_decision == matrix_result) else 'INCORRECT'

            samples.append(SuitabilityEvaluationSample(
                customer_id=f"C{i:05d}",
                product_id=f"P{i%10+1:03d}",
                customer_risk=c_risk,
                product_risk=p_risk,
                matrix_result=matrix_result,
                final_decision=final_decision,
                is_manual_review=manual_verdict is not None,
                manual_verdict=manual_verdict,
                should_allow=should_allow
            ))
        return samples

    @staticmethod
    def generate_intent_samples(n: int = 2000, online: bool = False) -> List[IntentEvaluationSample]:
        samples = []
        for i in range(n):
            # 模拟真实意图分与转化关系
            true_prob = np.random.beta(2, 3)  # 随机真实概率
            # 模型预测分数：在真实概率上加噪声
            score = np.clip(true_prob + np.random.normal(0, 0.1), 0, 1)
            # 转化标签：依据真实概率
            converted = 1 if np.random.rand() < true_prob else 0

            sample = IntentEvaluationSample(
                customer_id=f"C{i:05d}",
                feature_time=datetime.now(),
                intent_score=score,
                converted=converted,
                is_recommended_by_model=score > 0.5 if online else False,
                is_accepted_by_adviser=bool(np.random.rand() < 0.6) if online else False
            )
            samples.append(sample)
        return samples

    @staticmethod
    def generate_fusion_samples(n: int = 500) -> List[FusionEvaluationSample]:
        actions = ['PROACTIVE_CLOSING', 'NURTURE_CONTENT', 'LOW_PRIORITY', 'HUMAN_REVIEW', 'BLOCK']
        samples = []
        for i in range(n):
            action = np.random.choice(actions, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            # 不同动作转化率不同
            if action == 'PROACTIVE_CLOSING':
                conv_rate = 0.15
            elif action == 'NURTURE_CONTENT':
                conv_rate = 0.08
            elif action == 'LOW_PRIORITY':
                conv_rate = 0.02
            elif action == 'HUMAN_REVIEW':
                conv_rate = 0.10
            else:  # BLOCK
                conv_rate = 0.0

            converted = 1 if np.random.rand() < conv_rate else 0

            # 投诉率
            complaint_rate = 0.01 if action != 'BLOCK' else 0.005
            complaint = 1 if np.random.rand() < complaint_rate else 0

            # 合规复核（仅对BLOCK和HUMAN_REVIEW）
            compliance_pass = None
            if action in ['BLOCK', 'HUMAN_REVIEW']:
                # 假设通过率90%
                compliance_pass = 1 if np.random.rand() < 0.9 else 0

            samples.append(FusionEvaluationSample(
                customer_id=f"C{i:05d}",
                product_id=f"P{i%10+1:03d}",
                action=action,
                intent_score=np.random.uniform(0, 1),
                suitability_level='ALLOW' if action not in ['BLOCK','HUMAN_REVIEW'] else ('RESTRICTED' if action=='HUMAN_REVIEW' else 'FORBID'),
                converted=converted,
                complaint_received=complaint,
                compliance_review_passed=compliance_pass
            ))
        return samples


# ================================================================
# 4. 主程序演示
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(" Validity 效果评估体系 演示")
    print("=" * 80)

    # 生成模拟数据
    gen = EvaluationDataGenerator()
    suit_samples = gen.generate_suitability_samples(1500)
    intent_offline = gen.generate_intent_samples(3000, online=False)
    intent_online = gen.generate_intent_samples(2000, online=True)
    random_control = gen.generate_intent_samples(1000, online=True)  # 对照组
    fusion_samples = gen.generate_fusion_samples(800)

    # 初始化评估器
    evaluator = ValidityEvaluator(output_dir="./validity_reports")

    # 生成完整报告
    report = evaluator.generate_full_report(
        suitability_samples=suit_samples,
        intent_offline_samples=intent_offline,
        intent_online_samples=intent_online,
        fusion_samples=fusion_samples,
        random_control_samples=random_control,
        report_name="prudence_validity_report"
    )

    print("\n✅ 评估完成！报告已生成在 ./validity_reports/ 目录")
    print("   包含 JSON、Excel 和可视化图表（若依赖已安装）")