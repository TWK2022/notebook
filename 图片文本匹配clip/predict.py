# 官方项目：https://github.com/openai/CLIP
# 中文文本模型：https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese
# 2021年openai发布的模型，用于将图片和文字描述匹配
# 相当于1个图片分类模型与1个文本分类模型相结合，两个模型的标签一致，只是标签并非独热编码而是一段特征向量
# 因此一张图片经过图片模型得到的特征向量和这张图片的描述经过文本模型得到的特征向量相近，从而能够通过图片找文本，也可以通过文本找图片
# 原clip官方文本模型只支持英文，国内有人训练了中文的文本模型，只支持ViT-L/14(890M)
import clip
import torch
import argparse
import numpy as np
import transformers
import pandas as pd

parser = argparse.ArgumentParser(description='')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置|')
parser.add_argument('--model_name', default='ViT-L/14', type=str, help='|模型名称，中文文本模型只支持ViT-L/14(890M)|')
parser.add_argument('--english_score_threshold', default=20, type=int, help='|得分阈值|')
parser.add_argument('--chinese_score_threshold', default=15, type=int, help='|得分阈值|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
args = parser.parse_args()


class predict:
    def __init__(self, args):
        # 模型
        model, image_deal = clip.load(args.model_name, device=args.device)  # clip模型：图片模型+英文文本模型
        chinese_encode = transformers.BertForSequenceClassification.from_pretrained(
            "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval().half().to(args.device)  # 中文文本模型，只支持ViT-L/14(890M)
        print(f'| 模型加载成功:{args.model_name} | 中文文本模型:IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese |')
        self.model = model.eval()
        self.chinese_encode = chinese_encode
        # 数据
        df = pd.read_csv(args.database_path)
        column = df.columns
        image_feature = df.values
        self.column = column
        self.image_feature = image_feature
        # 阈值
        self.english_score_threshold = args.english_score_threshold
        self.chinese_score_threshold = args.chinese_score_threshold

    def _predict(self, english_text=None, chinese_text=None):
        with torch.no_grad():
            # 英文
            english_colunm = None
            if english_text:
                english_sequence = clip.tokenize(args.english_text).to(args.device)  # 处理
                english_text_feature = self.model.encode_text(english_sequence)  # 推理
                english_text_feature /= torch.norm(english_text_feature, dim=1, keepdim=True)  # 归一化
                english_text_feature = english_text_feature.cpu().numpy()
                english_colunm, english_score = self._deal(english_text_feature, self.english_score_threshold)
            # 中文
            chinese_colunm = None
            if chinese_text:
                chinese_tokenizer = transformers.BertTokenizer.from_pretrained(
                    "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
                chinese_sequence = chinese_tokenizer(args.chinese_text, max_length=77, padding='max_length',
                                                     truncation=True, return_tensors='pt')['input_ids'].type(
                    torch.int32).to(args.device)  # 处理
                chinese_text_feature = self.chinese_encode(chinese_sequence).logits  # 推理
                chinese_text_feature /= torch.norm(chinese_text_feature, dim=1, keepdim=True)  # 归一化
                chinese_text_feature = chinese_text_feature.cpu().numpy()
                chinese_colunm, chinese_score = self._deal(chinese_text_feature, self.chinese_score_threshold)
        return english_colunm, english_score, chinese_colunm, chinese_score

    def _deal(self, text_feature, score_threshold):
        score = 100.0 * np.dot(text_feature, self.image_feature)
        judge = np.where(score > score_threshold, True, False)
        colunm = [self.column[judge[_]].tolist() for _ in range(len(judge))]
        score = [score[_][judge[_]].tolist() for _ in range(len(judge))]
        return colunm, score


if __name__ == '__main__':
    # 文本
    args.english_text = ['lipstick', 'eyeliner']
    args.chinese_text = ['口红', '眼线笔']
    predictor = predict(args)
    english_colunm, english_score, chinese_colunm, chinese_score = \
        predictor._predict(args.english_text, args.chinese_text)  # 单个/多个文本
    print(f'| 英文:{args.english_text}:{english_colunm} |')
    print(f'| 得分:{args.english_text}:{english_score} |')
    print(f'| 中文:{args.chinese_text}:{chinese_colunm} |')
    print(f'| 得分:{args.chinese_text}:{chinese_score} |')
