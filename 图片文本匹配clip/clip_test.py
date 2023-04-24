# 官方项目：https://github.com/openai/CLIP
# 中文文本模型：https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese
# 2021年openai发布的模型，用于将图片和文字描述匹配
# 相当于1个图片分类模型与1个文本分类模型相结合，两个模型的标签一致，只是标签并非独热编码而是一段特征向量
# 因此一张图片经过图片模型得到的特征向量和这张图片的描述经过文本模型得到的特征向量相近，从而能够通过图片找文本，也可以通过文本找图片
# 原clip官方文本模型只支持英文，国内有人训练了中文的文本模型，只支持ViT-L/14(890M)
import os
import PIL
import clip
import torch
import transformers
import numpy as np

model_list = clip.available_models()
print(model_list)
# 模型
model_name = 'ViT-L/14'  # 有多种型号，但中文文本模型只支持ViT-L/14(890M)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, image_deal = clip.load(model_name, device=device)  # clip模型：图片模型+英文文本模型
chinese_encode = transformers.BertForSequenceClassification.from_pretrained(
    "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval().half().to(device)  # 中文文本模型，只支持ViT-L/14(890M)
print(f'| 加载模型成功:{model_name} | 中文文本模型:IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese |')
# 图片处理
image_path = 'image'
image_name = sorted(os.listdir(image_path))
print(f'| 图片:{image_name} |')
image_list = []
for i in range(len(image_name)):
    image = image_deal(PIL.Image.open(f'{image_path}/{image_name[i]}'))
    image_list.append(image)
image_bacth = torch.stack(image_list, dim=0).to(device)
print(f'| image_bacth.shape:{image_bacth.shape},{image_bacth.dtype} |')
# 文本处理
english_text = ['cat', 'a cat']
chinese_text = ['猫', '一只猫']
# 英文
english_sequence = clip.tokenize(english_text).to(device)
print(f'| english_sequence.shape:{english_sequence.shape},{english_sequence.dtype} |')
# 中文
chinese_tokenizer = transformers.BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
chinese_sequence = chinese_tokenizer(chinese_text, max_length=77, padding='max_length', truncation=True,
                                     return_tensors='pt')['input_ids'].type(torch.int32).to(device)
print(f'| chinese_sequence.shape:{chinese_sequence.shape},{chinese_sequence.dtype} |')
# 推理
with torch.no_grad():
    model = model.eval()
    # 图片特征
    image_feature = model.encode_image(image_bacth)
    image_feature /= torch.norm(image_feature, dim=1, keepdim=True)
    print(f'| image_feature.shape:{image_feature.shape},{image_feature.dtype} |')
    # 英文文本特征
    english_text_feature = model.encode_text(english_sequence)
    print(f'| english_text_feature.shape:{english_text_feature.shape},{english_text_feature.dtype} |')
    # 中文文本特征
    chinese_text_feature = chinese_encode(chinese_sequence).logits
    print(f'| chinese_text_feature.shape:{chinese_text_feature.shape},{chinese_text_feature.dtype} |')
    # 图片和英文文本匹配
    english_text_feature /= torch.norm(english_text_feature, dim=1, keepdim=True)
    score = (100.0 * english_text_feature @ image_feature.t()).softmax(dim=1)
    score = [[round(__.item(), 2) for __ in _] for _ in score]
    [print(f'英文模型:{english_text[_]}:{score[_]}') for _ in range(len(score))]
    # 图片和中文文本匹配
    chinese_text_feature /= torch.norm(chinese_text_feature, dim=1, keepdim=True)
    score = (100.0 * chinese_text_feature @ image_feature.t()).softmax(dim=1)
    score = [[round(__.item(), 2) for __ in _] for _ in score]
    [print(f'中文模型:{chinese_text[_]}:{score[_]}') for _ in range(len(score))]
