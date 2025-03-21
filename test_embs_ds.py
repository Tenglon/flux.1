import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk

# 假设这是你的原始数据集
original_dataset = load_from_disk("./Donghyun99/CUB-200-2011_latents.hf/")

# 1. 创建类别级别的属性（示例：从文件加载或其他来源）
num_classes = 200
class_attributes = np.random.random((num_classes, 300))  # 假设每个类别有512维属性

# 2. 创建类别级别的数据集
class_level_dataset = Dataset.from_dict({
    'class_id': list(range(num_classes)),
    'class_attributes': class_attributes.tolist()
})

# 3. 创建一个数据集字典来同时存储两种视图
dataset_dict = DatasetDict({
    'sample_level': original_dataset,  # 11788 行
    'class_level': class_level_dataset        # 200 行
})

# 使用示例
print("样本级数据集大小:", len(dataset_dict['sample_level']))  # 11788
print("类别级数据集大小:", len(dataset_dict['class_level']))   # 200

# 访问单个样本的属性
sample = dataset_dict['sample_level'][0]
# print("样本的类别:", sample['label'])
# print("样本的类别属性:", sample['class_attributes'])

# 访问类别级属性
class_info = dataset_dict['class_level'][0]
# print("类别ID:", class_info['class_id'])
# print("类别属性:", class_info['class_attributes'])

# 保存数据集
dataset_dict.save_to_disk("./local_datasets/test_embs_ds")

# 加载数据集
dataset_dict2 = load_from_disk("./local_datasets/test_embs_ds")

import pdb
pdb.set_trace()