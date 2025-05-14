#!/bin/bash

# 定义数据集数组
datasets=(
    "keremberke/pokemon-classification"
    "Donghyun99/CUB-200-2011"
    "Donghyun99/Stanford-Cars"
)

# 创建日志目录
mkdir -p logs

# 循环处理每个数据集
for dataset in "${datasets[@]}"; do
    # 提取数据集的简短名称作为作业名
    job_name=$(echo $dataset | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')
    
    # 创建临时作业脚本
    cat << EOF > job_${job_name}.sh
#!/bin/bash
#SBATCH --job-name=preprocess_${job_name}
#SBATCH --output=logs/${job_name}_%j.log
#SBATCH --error=logs/${job_name}_%j.log
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=24:00:00

source activate flux
srun python3 preprocess_dataset.py --dataset_name ${dataset}
EOF

    # 提交作业
    echo "提交任务 - ${dataset} 数据集预处理"
    sbatch job_${job_name}.sh
done

# 清理临时脚本文件（可选）
rm job_*.sh

# srun python3 preprocess_dataset.py --dataset_name "keremberke/pokemon-classification"
# srun python3 preprocess_dataset.py --dataset_name "Donghyun99/CUB-200-2011"
# srun python3 preprocess_dataset.py --dataset_name "Donghyun99/Stanford-Cars"