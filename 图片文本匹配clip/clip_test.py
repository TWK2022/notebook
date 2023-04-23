# 官方项目：https://github.com/openai/CLIP
# 中文文本模型：https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese
# 2021年openai发布的模型，用于将图片和文字描述匹配
# 相当于1个图片分类模型与1个文本分类模型相结合，两个模型的标签一致，只是标签并非独热编码而是一段特征向量
# 因此一张图片经过图片分类模型得到的特征向量和这张图片的描述经过文本模型得到的特征向量相近，从而能够通过图片找文本，也可以通过文本找图片
# 原clip官方文本模型只支持英文，国内有人训练了中文的文本模型，只支持ViT-L/14(890M)
import PIL
import clip
import torch
import transformers

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
a = image_deal(PIL.Image.open("image/01.jpg"))
b = image_deal(PIL.Image.open("image/02.jpg"))
image_bacth = torch.stack([a, b], dim=0).to(device)
print(f'| image_bacth.shape:{image_bacth.shape},{image_bacth.dtype} |')
# 文本处理
english_text = ['car', 'cat', 'a cat']
chinese_text = ['车', '猫', '一只猫']
# 英文
english_text = clip.tokenize(english_text).to(device)
print(f'| english_text.shape:{english_text.shape},{english_text.dtype} |')
# 中文
chinese_tokenizer = transformers.BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
chinese_text = chinese_tokenizer(chinese_text, max_length=77, padding='max_length', truncation=True,
                                 return_tensors='pt')['input_ids'].type(torch.int32).to(device)
print(f'| chinese_text.shape:{chinese_text.shape},{chinese_text.dtype} |')
# 推理
with torch.no_grad():
    model = model.eval()
    # 图片特征
    image_feature = model.encode_image(image_bacth)
    print(f'| image_feature.shape:{image_feature.shape},{image_feature.dtype} |')
    # 英文文本特征
    english_text_feature = model.encode_text(english_text)
    print(f'| english_text_feature.shape:{english_text_feature.shape},{english_text_feature.dtype} |')
    # 中文文本特征
    chinese_text_feature = chinese_encode(chinese_text).logits
    print(f'| chinese_text_feature.shape:{chinese_text_feature.shape},{chinese_text_feature.dtype} |')
    # 图片和英文文本匹配
    score = (image_feature @ english_text_feature.t())
    score = torch.softmax(score, dim=1)
    print("score(英文):", score)
    # 图片和中文文本匹配
    score = (image_feature @ chinese_text_feature.t())
    score = torch.softmax(score, dim=1)
    print("score(中文):", score)
