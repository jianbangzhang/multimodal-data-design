#!/usr/bin/env bash
# 公开数据集下载脚本
# 用法: bash scripts/download_data.sh [all|images|videos|audios]

set -e
MODE=${1:-all}

echo "========================================"
echo " 多模态数据集下载工具"
echo "========================================"

download_images() {
    echo ""
    echo "[图片数据集]"

    # COCO 2017 验证集（5K张，质量高，适合测试）
    echo ">> COCO 2017 val (5K张)..."
    mkdir -p data/raw/coco
    wget -nc -q --show-progress \
        http://images.cocodataset.org/zips/val2017.zip \
        -O data/raw/coco/val2017.zip
    unzip -q -n data/raw/coco/val2017.zip -d data/raw/coco/
    echo "   COCO annotations..."
    wget -nc -q --show-progress \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
        -O data/raw/coco/annotations.zip
    unzip -q -n data/raw/coco/annotations.zip -d data/raw/coco/

    # 复制到工作目录
    cp data/raw/coco/val2017/*.jpg data/raw/images/ 2>/dev/null || true
    echo "   ✓ COCO 图片已复制到 data/raw/images/"
}

download_videos() {
    echo ""
    echo "[视频数据集]"

    # UCF-101（13K视频，101个动作类别）
    echo ">> UCF-101 (需要手动申请，跳过自动下载)"
    echo "   申请地址: https://www.crcv.ucf.edu/data/UCF101.php"
    echo "   下载后放入: data/raw/videos/"

    # VideoChat2-IT（推荐直接用，2M条 SFT 格式）
    echo ""
    echo ">> VideoChat2-IT（推荐第一阶段直接使用）"
    echo "   huggingface-cli download OpenGVLab/VideoChat2-IT --local-dir data/videochat2"
}

download_audios() {
    echo ""
    echo "[音频数据集]"

    # AISHELL-1（178小时，中文 ASR 标准集）
    echo ">> AISHELL-1 (178小时中文)..."
    mkdir -p data/raw/audios
    wget -nc -q --show-progress \
        https://openslr.magicdatatech.com/resources/33/data_aishell.tgz \
        -O data/raw/data_aishell.tgz
    tar -xzf data/raw/data_aishell.tgz -C data/raw/audios/ 2>/dev/null || true
    echo "   ✓ AISHELL-1 解压完成"
}

download_sft_datasets() {
    echo ""
    echo "[第一阶段 SFT 数据集（下载即用）]"
    echo "以下数据集已经是 SFT 格式，无需标注，直接用于训练："
    echo ""
    echo "  LLaVA-Instruct-150K (通用图文问答):"
    echo "    huggingface-cli download liuhaotian/LLaVA-Instruct-150K --local-dir data/llava_instruct"
    echo ""
    echo "  ShareGPT4V-PT (高质量图像描述):"
    echo "    huggingface-cli download Lin-Chen/ShareGPT4V --local-dir data/sharegpt4v"
    echo ""
    echo "  LLaVA-OneVision (多任务多模态):"
    echo "    huggingface-cli download lmms-lab/LLaVA-OneVision-Data --local-dir data/llava_onevision"
}

case "$MODE" in
    images)  download_images ;;
    videos)  download_videos ;;
    audios)  download_audios ;;
    all)
        download_images
        download_videos
        download_audios
        download_sft_datasets
        ;;
    *)
        echo "用法: bash scripts/download_data.sh [all|images|videos|audios]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo " 下载完成"
echo "========================================"
