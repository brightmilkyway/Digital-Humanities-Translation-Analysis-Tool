#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译文本数字人文分析工具
Digital Humanities Translation Analysis Tool

功能：
1. 文本统计分析（词频、句长、段落结构等）
2. 翻译策略分析（直译vs意译倾向）
3. 文体风格对比
4. 语言复杂度分析
5. 可视化展示和报告生成

Author:袁禹豪&Claude Sonnet 4
"""

import re
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class TranslationAnalyzer:
    def __init__(self, source_file, target_file, source_lang='en', target_lang='zh'):
        """
        初始化翻译分析器
        
        Args:
            source_file: 原文文件路径
            target_file: 译文文件路径
            source_lang: 源语言代码
            target_lang: 目标语言代码
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 读取文本
        with open(source_file, 'r', encoding='utf-8') as f:
            self.source_text = f.read()
        with open(target_file, 'r', encoding='utf-8') as f:
            self.target_text = f.read()
        
        # 初始化分析结果存储
        self.analysis_results = {}
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def basic_statistics(self):
        """基础文本统计"""
        print("正在进行基础统计分析...")
        
        # 英文统计
        source_sentences = sent_tokenize(self.source_text)
        source_words = word_tokenize(self.source_text.lower())
        source_words = [w for w in source_words if w.isalpha()]
        
        # 中文统计
        target_sentences = re.split(r'[。！？；]', self.target_text)
        target_sentences = [s.strip() for s in target_sentences if s.strip()]
        target_words = list(jieba.cut(self.target_text))
        target_words = [w.strip() for w in target_words if w.strip() and len(w) > 1]
        
        stats = {
            'source': {
                'characters': len(self.source_text),
                'words': len(source_words),
                'sentences': len(source_sentences),
                'paragraphs': len([p for p in self.source_text.split('\n\n') if p.strip()]),
                'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in source_sentences]),
                'avg_word_length': np.mean([len(w) for w in source_words])
            },
            'target': {
                'characters': len(self.target_text),
                'words': len(target_words),
                'sentences': len(target_sentences),
                'paragraphs': len([p for p in self.target_text.split('\n\n') if p.strip()]),
                'avg_sentence_length': np.mean([len(list(jieba.cut(s))) for s in target_sentences if s]),
                'avg_word_length': np.mean([len(w) for w in target_words])
            }
        }
        
        # 计算翻译比率
        stats['ratios'] = {
            'character_ratio': stats['target']['characters'] / stats['source']['characters'],
            'word_ratio': stats['target']['words'] / stats['source']['words'],
            'sentence_ratio': stats['target']['sentences'] / stats['source']['sentences']
        }
        
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def lexical_analysis(self):
        """词汇分析"""
        print("正在进行词汇分析...")
        
        # 英文词汇分析
        source_words = word_tokenize(self.source_text.lower())
        source_words = [w for w in source_words if w.isalpha()]
        stop_words = set(stopwords.words('english'))
        source_content_words = [w for w in source_words if w not in stop_words]
        
        # 中文词汇分析
        target_words = list(jieba.cut(self.target_text))
        target_words = [w.strip() for w in target_words if w.strip() and len(w) > 1]
        
        # 词频统计
        source_freq = Counter(source_content_words)
        target_freq = Counter(target_words)
        
        # 词汇丰富度（TTR - Type Token Ratio）
        source_ttr = len(set(source_content_words)) / len(source_content_words) if source_content_words else 0
        target_ttr = len(set(target_words)) / len(target_words) if target_words else 0
        
        lexical_data = {
            'source_top_words': source_freq.most_common(20),
            'target_top_words': target_freq.most_common(20),
            'source_ttr': source_ttr,
            'target_ttr': target_ttr,
            'source_unique_words': len(set(source_content_words)),
            'target_unique_words': len(set(target_words))
        }
        
        self.analysis_results['lexical'] = lexical_data
        return lexical_data
    
    def complexity_analysis(self):
        """复杂度分析"""
        print("正在进行复杂度分析...")
        
        # 英文可读性指标
        source_flesch = flesch_reading_ease(self.source_text)
        source_fk_grade = flesch_kincaid_grade(self.source_text)
        
        # 句子长度分布
        source_sentences = sent_tokenize(self.source_text)
        target_sentences = re.split(r'[。！？；]', self.target_text)
        target_sentences = [s.strip() for s in target_sentences if s.strip()]
        
        source_sent_lengths = [len(word_tokenize(s)) for s in source_sentences]
        target_sent_lengths = [len(list(jieba.cut(s))) for s in target_sentences if s]
        
        complexity_data = {
            'source_flesch_score': source_flesch,
            'source_fk_grade': source_fk_grade,
            'source_sent_lengths': source_sent_lengths,
            'target_sent_lengths': target_sent_lengths,
            'source_avg_sent_length': np.mean(source_sent_lengths),
            'target_avg_sent_length': np.mean(target_sent_lengths),
            'source_sent_length_std': np.std(source_sent_lengths),
            'target_sent_length_std': np.std(target_sent_lengths)
        }
        
        self.analysis_results['complexity'] = complexity_data
        return complexity_data
    
    def translation_strategy_analysis(self):
        """翻译策略分析"""
        print("正在进行翻译策略分析...")
        
        # 基于句子对齐进行分析
        source_sentences = sent_tokenize(self.source_text)
        target_sentences = re.split(r'[。！？；]', self.target_text)
        target_sentences = [s.strip() for s in target_sentences if s.strip()]
        
        # 简单的句子长度比较来推断翻译策略
        length_ratios = []
        min_len = min(len(source_sentences), len(target_sentences))
        
        for i in range(min_len):
            source_len = len(word_tokenize(source_sentences[i]))
            target_len = len(list(jieba.cut(target_sentences[i])))
            if source_len > 0:
                ratio = target_len / source_len
                length_ratios.append(ratio)
        
        # 翻译策略指标
        avg_ratio = np.mean(length_ratios) if length_ratios else 1
        ratio_variance = np.var(length_ratios) if length_ratios else 0
        
        # 根据比率推断翻译倾向
        if avg_ratio > 1.2:
            strategy_tendency = "意译倾向（扩展性翻译）"
        elif avg_ratio < 0.8:
            strategy_tendency = "直译倾向（压缩性翻译）"
        else:
            strategy_tendency = "平衡翻译策略"
        
        strategy_data = {
            'length_ratios': length_ratios,
            'avg_length_ratio': avg_ratio,
            'ratio_variance': ratio_variance,
            'strategy_tendency': strategy_tendency,
            'sentence_alignment_quality': min_len / max(len(source_sentences), len(target_sentences))
        }
        
        self.analysis_results['strategy'] = strategy_data
        return strategy_data
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("正在生成可视化图表...")
        
        # 创建图表目录
        import os
        if not os.path.exists('translation_analysis_charts'):
            os.makedirs('translation_analysis_charts')
        
        # 1. 基础统计对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 文本量对比
        stats = self.analysis_results['basic_stats']
        categories = ['字符数', '词数', '句数', '段落数']
        source_values = [stats['source']['characters'], stats['source']['words'], 
                        stats['source']['sentences'], stats['source']['paragraphs']]
        target_values = [stats['target']['characters'], stats['target']['words'], 
                        stats['target']['sentences'], stats['target']['paragraphs']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0,0].bar(x - width/2, source_values, width, label='原文', alpha=0.8)
        axes[0,0].bar(x + width/2, target_values, width, label='译文', alpha=0.8)
        axes[0,0].set_title('文本量统计对比')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(categories)
        axes[0,0].legend()
        
        # 2. 句长分布对比
        complexity = self.analysis_results['complexity']
        axes[0,1].hist(complexity['source_sent_lengths'], bins=20, alpha=0.7, label='原文句长')
        axes[0,1].hist(complexity['target_sent_lengths'], bins=20, alpha=0.7, label='译文句长')
        axes[0,1].set_title('句子长度分布')
        axes[0,1].set_xlabel('句子长度（词数）')
        axes[0,1].set_ylabel('频次')
        axes[0,1].legend()
        
        # 3. TTR对比
        lexical = self.analysis_results['lexical']
        ttr_data = ['原文TTR', '译文TTR']
        ttr_values = [lexical['source_ttr'], lexical['target_ttr']]
        axes[1,0].bar(ttr_data, ttr_values, color=['skyblue', 'lightcoral'])
        axes[1,0].set_title('词汇丰富度对比（TTR）')
        axes[1,0].set_ylabel('TTR值')
        
        # 4. 翻译策略分析
        if 'strategy' in self.analysis_results:
            strategy = self.analysis_results['strategy']
            if strategy['length_ratios']:
                axes[1,1].hist(strategy['length_ratios'], bins=15, alpha=0.7, color='green')
                axes[1,1].axvline(strategy['avg_length_ratio'], color='red', linestyle='--', 
                                label=f'平均比率: {strategy["avg_length_ratio"]:.2f}')
                axes[1,1].set_title('句长比率分布（译文/原文）')
                axes[1,1].set_xlabel('长度比率')
                axes[1,1].set_ylabel('频次')
                axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('translation_analysis_charts/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. 词云图
        self._create_wordclouds()
        
        # 6. 交互式图表
        self._create_interactive_charts()
    
    def _create_wordclouds(self):
        """创建词云图"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 原文词云
        source_words = word_tokenize(self.source_text.lower())
        source_words = [w for w in source_words if w.isalpha() and len(w) > 3]
        stop_words = set(stopwords.words('english'))
        source_words = [w for w in source_words if w not in stop_words]
        source_text_clean = ' '.join(source_words)
        
        if source_text_clean:
            wordcloud_en = WordCloud(width=800, height=400, background_color='white').generate(source_text_clean)
            axes[0].imshow(wordcloud_en, interpolation='bilinear')
            axes[0].set_title('原文词云图', fontsize=16)
            axes[0].axis('off')
        
        # 译文词云
        target_words = jieba.cut(self.target_text)
        target_words = [w for w in target_words if len(w) > 1 and w.strip()]
        target_text_clean = ' '.join(target_words)
        
        if target_text_clean:
            wordcloud_zh = WordCloud(font_path='simhei.ttf', width=800, height=400, 
                                   background_color='white').generate(target_text_clean)
            axes[1].imshow(wordcloud_zh, interpolation='bilinear')
            axes[1].set_title('译文词云图', fontsize=16)
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('translation_analysis_charts/wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_interactive_charts(self):
        """创建交互式图表"""
        # 词频对比图
        lexical = self.analysis_results['lexical']
        
        source_words, source_freqs = zip(*lexical['source_top_words'][:15])
        target_words, target_freqs = zip(*lexical['target_top_words'][:15])
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('原文高频词', '译文高频词'))
        
        fig.add_trace(go.Bar(x=list(source_words), y=list(source_freqs), name='原文'), row=1, col=1)
        fig.add_trace(go.Bar(x=list(target_words), y=list(target_freqs), name='译文'), row=1, col=2)
        
        fig.update_layout(title_text="高频词对比", showlegend=False)
        fig.write_html('translation_analysis_charts/word_frequency_comparison.html')
    
    def generate_report(self):
        """生成分析报告"""
        print("正在生成分析报告...")
        
        report = f"""译文本数字人文分析报告
Translation Digital Humanities Analysis Report

1. 基础统计分析

原文统计：
- 字符数：{self.analysis_results['basic_stats']['source']['characters']:,}
- 词数：{self.analysis_results['basic_stats']['source']['words']:,}
- 句数：{self.analysis_results['basic_stats']['source']['sentences']:,}
- 段落数：{self.analysis_results['basic_stats']['source']['paragraphs']:,}
- 平均句长：{self.analysis_results['basic_stats']['source']['avg_sentence_length']:.2f} 词/句
- 平均词长：{self.analysis_results['basic_stats']['source']['avg_word_length']:.2f} 字符/词

译文统计：
- 字符数：{self.analysis_results['basic_stats']['target']['characters']:,}
- 词数：{self.analysis_results['basic_stats']['target']['words']:,}
- 句数：{self.analysis_results['basic_stats']['target']['sentences']:,}
- 段落数：{self.analysis_results['basic_stats']['target']['paragraphs']:,}
- 平均句长：{self.analysis_results['basic_stats']['target']['avg_sentence_length']:.2f} 词/句
- 平均词长：{self.analysis_results['basic_stats']['target']['avg_word_length']:.2f} 字符/词

翻译比率：
- 字符比率：{self.analysis_results['basic_stats']['ratios']['character_ratio']:.2f}
- 词汇比率：{self.analysis_results['basic_stats']['ratios']['word_ratio']:.2f}
- 句子比率：{self.analysis_results['basic_stats']['ratios']['sentence_ratio']:.2f}

2. 词汇分析

词汇丰富度（TTR）：
- 原文TTR：{self.analysis_results['lexical']['source_ttr']:.4f}
- 译文TTR：{self.analysis_results['lexical']['target_ttr']:.4f}

独特词汇数：
- 原文：{self.analysis_results['lexical']['source_unique_words']:,}
- 译文：{self.analysis_results['lexical']['target_unique_words']:,}

3. 复杂度分析

原文可读性：
- Flesch可读性得分：{self.analysis_results['complexity']['source_flesch_score']:.2f}
- Flesch-Kincaid等级：{self.analysis_results['complexity']['source_fk_grade']:.2f}

句子复杂度：
- 原文平均句长：{self.analysis_results['complexity']['source_avg_sent_length']:.2f} ± {self.analysis_results['complexity']['source_sent_length_std']:.2f}
- 译文平均句长：{self.analysis_results['complexity']['target_avg_sent_length']:.2f} ± {self.analysis_results['complexity']['target_sent_length_std']:.2f}

4. 翻译策略分析

"""
        
        if 'strategy' in self.analysis_results:
            strategy = self.analysis_results['strategy']
            report += f"""翻译策略倾向： {strategy['strategy_tendency']}

关键指标：
- 平均长度比率：{strategy['avg_length_ratio']:.2f}
- 比率方差：{strategy['ratio_variance']:.4f}
- 句子对齐质量：{strategy['sentence_alignment_quality']:.2f}

5. 分析结论

基于以上数据分析，您的翻译呈现以下特点：

1. 文本扩展性：译文相对原文的字符比率为 {self.analysis_results['basic_stats']['ratios']['character_ratio']:.2f}，表明翻译过程中文本长度{'增加' if self.analysis_results['basic_stats']['ratios']['character_ratio'] > 1 else '减少'}了 {abs(self.analysis_results['basic_stats']['ratios']['character_ratio'] - 1) * 100:.1f}%

2. 词汇丰富度：译文TTR为 {self.analysis_results['lexical']['target_ttr']:.4f}，{'高于' if self.analysis_results['lexical']['target_ttr'] > self.analysis_results['lexical']['source_ttr'] else '低于'}原文的 {self.analysis_results['lexical']['source_ttr']:.4f}

3.*翻译策略：{strategy['strategy_tendency']}，平均句长比率为 {strategy['avg_length_ratio']:.2f}

4. 可读性：原文Flesch得分为 {self.analysis_results['complexity']['source_flesch_score']:.1f}，属于{'容易' if self.analysis_results['complexity']['source_flesch_score'] > 70 else '中等' if self.analysis_results['complexity']['source_flesch_score'] > 30 else '困难'}阅读水平

6. 翻译质量评估建议

- 考虑句子长度的均衡性，避免过长或过短的句子
- 注意保持适当的词汇丰富度
- 根据文本类型调整翻译策略（直译vs意译）
- 关注译文的流畅性和可读性

---
本报告由数字人文翻译分析工具生成
"""
        
        # 保存报告
        with open('translation_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("分析报告已保存为 'translation_analysis_report.md'")
        return report
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("开始翻译文本数字人文分析...")
        print("=" * 50)
        
        # 执行各项分析
        self.basic_statistics()
        self.lexical_analysis()
        self.complexity_analysis()
        self.translation_strategy_analysis()
        
        # 生成可视化
        self.create_visualizations()
        
        # 生成报告
        report = self.generate_report()
        
        print("=" * 50)
        print("分析完成！生成的文件：")
        print("- translation_analysis_report.md (分析报告)")
        print("- translation_analysis_charts/ (可视化图表)")
        print("  ├── comprehensive_analysis.png")
        print("  ├── wordclouds.png")
        print("  └── word_frequency_comparison.html")
        
        return self.analysis_results


def main():
    """主函数 - 使用示例"""
    print("=" * 50)
    print("🔍 翻译文本数字人文分析工具")
    print("   Translation Digital Humanities Analysis Tool")
    print("=" * 50)
    
    # ========== 在这里直接指定文件路径 ==========
    source_file = r"E:\南开\英语专业\文学翻译\期末项目\source.txt"     # 原文文件路径
    target_file = r"E:\南开\英语专业\文学翻译\期末项目\target.txt"     # 译文文件路径
    # ==========================================
    
    print(f"📁 分析文件：")
    print(f"   原文文件: {source_file}")
    print(f"   译文文件: {target_file}")
    
    try:
        print("\n🚀 开始分析...")
        
        # 创建分析器实例
        analyzer = TranslationAnalyzer(source_file, target_file)
        
        # 运行完整分析
        results = analyzer.run_full_analysis()
        
        print("\n" + "=" * 50)
        print("📊 分析结果预览：")
        print(f"   原文词数: {results['basic_stats']['source']['words']:,}")
        print(f"   译文词数: {results['basic_stats']['target']['words']:,}")
        print(f"   词汇比率: {results['basic_stats']['ratios']['word_ratio']:.2f}")
        print(f"   翻译策略: {results['strategy']['strategy_tendency']}")
        print("=" * 50)
        print("✨ 分析完成！请查看生成的报告和图表文件。")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误：找不到文件")
        print(f"   请确保以下文件存在：")
        print(f"   - {source_file}")
        print(f"   - {target_file}")
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误：{str(e)}")
        print("请检查文件格式是否正确（应为UTF-8编码的文本文件）")


if __name__ == "__main__":
    main()
