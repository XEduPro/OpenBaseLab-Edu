# -*- coding: utf-8 -*-
"""
BaseDT NPZ 数据集制作使用示例。
演示如何从视频或 CSV 生成适用于 RNN/序列分类的 NPZ 数据集。
"""

import os
import sys
import shutil
import tempfile

# 优先加载本地 XEdu-python-main，避免使用 site-packages 的旧版本
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_examples_dir))  # XEdu-python-main
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from XEdu.hub.BaseDT import make_npz_dataset, NPZGenerator


def demo_video_to_npz(video_path, output_path='dataset_from_video.npz'):
    """
    将单个视频制作为 NPZ 数据集。
    视频模式需要按类别分目录，单视频时自动创建临时目录结构（单类）。
    """
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        print('视频不存在:', video_path)
        return
    tmp_dir = tempfile.mkdtemp(prefix='npz_video_')
    class_dir = os.path.join(tmp_dir, 'single')
    os.makedirs(class_dir)
    dst = os.path.join(class_dir, os.path.basename(video_path))
    shutil.copy(video_path, dst)
    print('临时目录:', tmp_dir)
    make_npz_dataset(
        tmp_dir,
        output_path,
        data_type='video',
        pose_source='xedu',
        sequence_length=30,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print('视频 -> NPZ 完成，输出:', output_path)
    import numpy as np
    data = np.load(output_path)
    print('data.shape:', data['data'].shape, 'label.shape:', data['label'].shape)


def demo_iris_csv_to_npz(csv_path, output_path='dataset_from_iris.npz'):
    """将 iris.csv 制作为 NPZ 数据集。"""
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        print('CSV 不存在:', csv_path)
        return
    make_npz_dataset(
        csv_path,
        output_path,
        data_type='csv',
        sequence_length=10,
        label_column=-1,
        delimiter=',',
        skiprows=1,
    )
    print('CSV -> NPZ 完成，输出:', output_path)
    import numpy as np
    data = np.load(output_path)
    print('data.shape:', data['data'].shape, 'label.shape:', data['label'].shape)


def demo_make_npz_from_csv():
    """
    示例1：从 CSV 表格数据生成 NPZ 数据集。
    CSV 格式：每行一个时间步，最后一列为类别标签。按类别分组后切成 sequence_length 长度的序列。
    """
    import numpy as np

    # 创建示例 CSV 数据（若 data.csv 不存在）
    csv_path = 'data_for_npz_demo.csv'
    if not os.path.exists(csv_path):
        # 3 类，每类 100 行，5 个特征列 + 1 个标签列
        np.random.seed(42)
        rows = []
        header = 'f1,f2,f3,f4,f5,label'
        for label in range(3):
            for _ in range(100):
                feat = np.random.randn(5).astype(float)
                rows.append(','.join(map(str, feat)) + ',' + str(label))
        with open(csv_path, 'w') as f:
            f.write(header + '\n' + '\n'.join(rows))
        print('已创建示例 CSV:', csv_path)

    output_path = 'dataset_from_csv.npz'
    make_npz_dataset(
        csv_path,
        output_path,
        data_type='csv',
        sequence_length=10,
        label_column=-1,
        delimiter=',',
        skiprows=1,
    )
    print('CSV -> NPZ 完成，输出:', output_path)

    # 验证
    data = np.load(output_path)
    print('data.shape:', data['data'].shape, 'label.shape:', data['label'].shape)


def demo_make_npz_from_video():
    """
    示例2：从视频目录生成 NPZ 数据集。
    目录结构：dataset_path/类别名/视频文件
    默认使用 XEdu det_body+pose_body26（52维/帧），不需额外模型。也可用 pose_source='mediapipe' 需 .task 模型。
    """
    video_dir = './video'  # 视频目录：video/waving/*.mp4, video/walking/*.mp4 等

    if not os.path.isdir(video_dir):
        print('视频目录不存在:', video_dir, '请准备按类别分文件夹的视频目录。')
        return

    # 默认 pose_source='xedu'，无需 model_path
    make_npz_dataset(
        video_dir,
        'dataset_from_video.npz',
        data_type='video',
        pose_source='xedu',  # 默认，XEdu det_body+pose_body26
        sequence_length=30,
    )
    print('视频 -> NPZ 完成，输出: dataset_from_video.npz')

    # 若使用 MediaPipe（132维/帧），需传入 model_path
    # make_npz_dataset(video_dir, 'dataset_mediapipe.npz', data_type='video',
    #                 pose_source='mediapipe', model_path='pose_landmarker_full.task')


def demo_npz_generator_class():
    """
    示例3：使用 NPZGenerator 类（兼容原 npz_generator 接口）。
    默认 pose_source='xedu' 不需 model_path；pose_source='mediapipe' 需 model_path。
    """
    video_dir = './video'

    if not os.path.isdir(video_dir):
        print('请准备视频目录。')
        return

    # 默认使用 XEdu，不需 model_path
    gen = NPZGenerator(
        dataset_path=video_dir,
        sequence_length=30,
        pose_source='xedu',  # 默认
    )
    gen.generate_dataset('dataset.npz')
    print('标签映射:', gen.get_label_map())
    print('标签列表:', gen.get_label_map_list())

    # 推理：对单个视频生成推理用数组
    # inf_data = gen.generate_for_inference('test.mp4')
    # 设置标签名用于解析推理结果
    gen.set_label_map_list(['waving', 'walking', 'stretching'])
    # gen.see_result(model_output)  # 解析并打印推理结果


def demo_auto_detect():
    """
    示例4：自动检测数据类型（目录->视频，.csv->CSV）。
    """
    # CSV 示例
    csv_path = 'data_for_npz_demo.csv'
    if os.path.exists(csv_path):
        make_npz_dataset(csv_path, 'auto_csv.npz', data_type='auto', sequence_length=10)
        print('自动检测 CSV 完成')

    # 视频目录示例（默认 xedu 不需 model_path）
    if os.path.isdir('./video'):
        make_npz_dataset('./video', 'auto_video.npz', data_type='auto')
        print('自动检测视频完成')


if __name__ == '__main__':
    print('=' * 50)
    print('BaseDT NPZ 数据集制作示例')
    print('=' * 50)

    # 检查效果：指定路径制作 NPZ
    video_path = r'D:\Download\test (1).mp4'
    iris_csv_path = r'D:\XEdu\datasets\baseml\iris\iris.csv'

    print('\n--- 1. 视频转 NPZ ---')
    demo_video_to_npz(video_path, 'dataset_video.npz')

    print('\n--- 2. Iris CSV 转 NPZ ---')
    demo_iris_csv_to_npz(iris_csv_path, 'dataset_iris.npz')

    # 以下为其他示例（可选）
    # print('\n--- 示例: 从 CSV 生成 NPZ ---')
    # demo_make_npz_from_csv()
    # print('\n--- 示例: 从视频目录生成 NPZ ---')
    # demo_make_npz_from_video()
