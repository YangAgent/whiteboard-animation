#!/usr/bin/env python3
"""
白板手绘动画生成器
输入一张彩色图片，生成包含线稿绘制和上色两个阶段的白板手绘动画视频。
"""
import argparse
import os
import sys
import math
import time
import datetime
import cv2
import numpy as np
from pathlib import Path

# === 素材路径（相对于脚本位置） ===
_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_ASSETS_DIR = _SCRIPT_DIR.parent / "assets"
HAND_PATH = str(_ASSETS_DIR / "drawing-hand.png")

# === 固定算法参数 ===
FRAME_RATE = 60
SPLIT_LEN = 10
END_IMG_DURATION = 2
MAX_1080P = True
DEFAULT_DURATION = 10
SKIP_RATE = 4
HAND_TARGET_HT = 493  # 手部素材缩放到的目标高度（基于 1080p 画布的最佳尺寸）


# === 核心函数 ===

def euc_dist(arr1, point):
    square_sub = (arr1 - point) ** 2
    return np.sqrt(np.sum(square_sub, axis=1))


def get_extreme_coordinates(mask):
    indices = np.where(mask > 0)
    x = indices[1]
    y = indices[0]
    topleft = (np.min(x), np.min(y))
    bottomright = (np.max(x), np.max(y))
    return topleft, bottomright


def preprocess_image(img, variables):
    img = cv2.resize(img, (variables["resize_wd"], variables["resize_ht"]))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    variables["img_gray"] = img_gray
    variables["img_thresh"] = img_thresh
    variables["img"] = img
    return variables


def preprocess_hand_image(hand_path, variables):
    hand_rgba = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    if hand_rgba.shape[2] == 4:
        # 透明背景 PNG：直接从 alpha 通道提取蒙版
        hand_mask = hand_rgba[:, :, 3]
        hand = hand_rgba[:, :, :3]
    else:
        # 无 alpha 通道的回退：用白色背景检测
        hand = hand_rgba
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        _, hand_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    # 裁剪到有效区域
    top_left, bottom_right = get_extreme_coordinates(hand_mask)
    hand = hand[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hand_mask = hand_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # 按高度缩放到固定目标尺寸，宽度等比跟随
    hand_scale = HAND_TARGET_HT / hand.shape[0]
    new_ht = HAND_TARGET_HT
    new_wd = max(1, int(hand.shape[1] * hand_scale))
    interp = cv2.INTER_AREA if hand_scale < 1 else cv2.INTER_LINEAR
    hand = cv2.resize(hand, (new_wd, new_ht), interpolation=interp)
    hand_mask = cv2.resize(hand_mask, (new_wd, new_ht), interpolation=interp)
    # 归一化蒙版到 0.0~1.0，保留半透明边缘的平滑过渡
    hand_mask = hand_mask.astype(np.float32) / 255.0
    hand_mask_inv = 1.0 - hand_mask
    # 预乘：蒙版外区域置黑
    hand_bg_ind = np.where(hand_mask == 0)
    hand[hand_bg_ind] = [0, 0, 0]
    hand_ht, hand_wd = hand.shape[0], hand.shape[1]
    variables["hand_ht"] = hand_ht
    variables["hand_wd"] = hand_wd
    variables["hand"] = hand
    variables["hand_mask"] = hand_mask
    variables["hand_mask_inv"] = hand_mask_inv
    return variables


def draw_hand_on_img(drawing, hand, x, y, hand_mask, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd):
    remaining_ht = img_ht - y
    remaining_wd = img_wd - x
    crop_hand_ht = min(remaining_ht, hand_ht)
    crop_hand_wd = min(remaining_wd, hand_wd)
    if crop_hand_ht <= 0 or crop_hand_wd <= 0:
        return drawing
    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_cropped = hand_mask[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]
    for c in range(3):
        drawing[y:y + crop_hand_ht, x:x + crop_hand_wd, c] = (
            drawing[y:y + crop_hand_ht, x:x + crop_hand_wd, c] * hand_mask_inv_cropped
            + hand_cropped[:, :, c] * hand_mask_cropped
        )
    return drawing


def draw_masked_object(variables, target_cells, skip_rate=SKIP_RATE):
    img_thresh_copy = variables["img_thresh"].copy()
    split_len = variables["split_len"]
    resize_ht = variables["resize_ht"]
    resize_wd = variables["resize_wd"]

    n_cuts_vertical = int(math.ceil(resize_ht / split_len))
    n_cuts_horizontal = int(math.ceil(resize_wd / split_len))

    grid_of_cuts = np.array(np.split(img_thresh_copy, n_cuts_horizontal, axis=-1))
    grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))

    cut_having_black = (grid_of_cuts < 10) * 1
    cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
    cut_black_indices = np.array(np.where(cut_having_black > 0)).T
    actual_cells = len(cut_black_indices)
    target_frames = target_cells // skip_rate if skip_rate > 0 else actual_cells
    print(f"  网格总数: {n_cuts_vertical}x{n_cuts_horizontal}, 有内容的格子: {actual_cells}, 目标帧: {target_frames}")

    selected_ind = 0
    counter = 0
    frame_accumulator = 0.0
    frames_written = 0
    frame_ratio = target_frames / actual_cells if actual_cells > 0 else 1.0
    while len(cut_black_indices) > 1:
        selected_ind_val = cut_black_indices[selected_ind].copy()
        range_v_start = selected_ind_val[0] * split_len
        range_v_end = range_v_start + split_len
        range_h_start = selected_ind_val[1] * split_len
        range_h_end = range_h_start + split_len

        temp_drawing = np.zeros((split_len, split_len, 3))
        temp_drawing[:, :, 0] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
        temp_drawing[:, :, 1] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
        temp_drawing[:, :, 2] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]

        variables["drawn_frame"][range_v_start:range_v_end, range_h_start:range_h_end] = temp_drawing

        if variables["draw_hand"]:
            hand_coord_x = range_h_start + int(split_len / 2)
            hand_coord_y = range_v_start + int(split_len / 2)
            drawn_frame_with_hand = draw_hand_on_img(
                variables["drawn_frame"].copy(),
                variables["hand"].copy(),
                hand_coord_x, hand_coord_y,
                variables["hand_mask"].copy(),
                variables["hand_mask_inv"].copy(),
                variables["hand_ht"], variables["hand_wd"],
                resize_ht, resize_wd,
            )
        else:
            drawn_frame_with_hand = variables["drawn_frame"].copy()

        cut_black_indices[selected_ind] = cut_black_indices[-1]
        cut_black_indices = cut_black_indices[:-1]

        euc_arr = euc_dist(cut_black_indices, selected_ind_val)
        selected_ind = np.argmin(euc_arr)

        counter += 1
        frame_accumulator += frame_ratio
        n_frames = int(frame_accumulator) - frames_written
        if n_frames > 0:
            frame = drawn_frame_with_hand.astype(np.uint8)
            for _ in range(n_frames):
                variables["video_object"].write(frame)
            frames_written += n_frames

        if counter % 100 == 0:
            pct = int(counter / actual_cells * 100)
            print(f"  进度: {pct}% ({counter}/{actual_cells})")

    print(f"  绘制完成，共 {counter} 步, 写入 {frames_written} 帧")


def _build_brush_mask(radius):
    """预生成一个圆形笔刷蒙版，边缘高斯羽化，值域 0.0~1.0"""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    dist = np.sqrt(x * x + y * y).astype(np.float32)
    mask = np.clip(1.0 - (dist - radius * 0.75) / (radius * 0.25), 0, 1)
    return mask


def _apply_brush(drawn_frame, color_img, cx, cy, brush_mask, radius):
    """在 (cx, cy) 处用圆形笔刷将 drawn_frame 混合为 color_img"""
    h, w = drawn_frame.shape[:2]
    size = radius * 2 + 1

    y1 = max(cy - radius, 0)
    y2 = min(cy + radius + 1, h)
    x1 = max(cx - radius, 0)
    x2 = min(cx + radius + 1, w)

    by1 = y1 - (cy - radius)
    by2 = size - ((cy + radius + 1) - y2)
    bx1 = x1 - (cx - radius)
    bx2 = size - ((cx + radius + 1) - x2)

    mask_region = brush_mask[by1:by2, bx1:bx2]

    for c in range(3):
        drawn_frame[y1:y2, x1:x2, c] = (
            drawn_frame[y1:y2, x1:x2, c] * (1.0 - mask_region) +
            color_img[y1:y2, x1:x2, c] * mask_region
        )


def colorize_animation(variables, target_cells, skip_rate=SKIP_RATE, brush_radius=50):
    """
    第二阶段：上色。沿素描阶段同样的路径上色，手跟随移动。
    用圆形笔刷+羽化边缘替代矩形格子，模拟真实画笔上色效果。
    """
    img_thresh_copy = variables["img_thresh"].copy()
    split_len = variables["split_len"]
    resize_ht = variables["resize_ht"]
    resize_wd = variables["resize_wd"]
    color_img = variables["img"].astype(np.float32)

    variables["drawn_frame"] = variables["drawn_frame"].astype(np.float32)

    n_cuts_vertical = int(math.ceil(resize_ht / split_len))
    n_cuts_horizontal = int(math.ceil(resize_wd / split_len))

    grid_of_cuts = np.array(np.split(img_thresh_copy, n_cuts_horizontal, axis=-1))
    grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))

    cut_having_black = (grid_of_cuts < 10) * 1
    cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
    cut_black_indices = np.array(np.where(cut_having_black > 0)).T
    actual_cells = len(cut_black_indices)
    target_frames = target_cells // skip_rate if skip_rate > 0 else actual_cells

    brush_mask = _build_brush_mask(brush_radius)
    print(f"  上色格子数: {actual_cells} (笔刷半径: {brush_radius}px, 目标帧: {target_frames})")

    selected_ind = 0
    counter = 0
    frame_accumulator = 0.0
    frames_written = 0
    frame_ratio = target_frames / actual_cells if actual_cells > 0 else 1.0
    while len(cut_black_indices) > 1:
        selected_ind_val = cut_black_indices[selected_ind].copy()
        r, c = selected_ind_val[0], selected_ind_val[1]

        cx = c * split_len + split_len // 2
        cy = r * split_len + split_len // 2

        _apply_brush(variables["drawn_frame"], color_img,
                     cx, cy, brush_mask, brush_radius)

        if variables["draw_hand"]:
            drawn_frame_with_hand = draw_hand_on_img(
                variables["drawn_frame"].copy().astype(np.uint8),
                variables["hand"].copy(),
                cx, cy,
                variables["hand_mask"].copy(),
                variables["hand_mask_inv"].copy(),
                variables["hand_ht"], variables["hand_wd"],
                resize_ht, resize_wd,
            )
        else:
            drawn_frame_with_hand = (
                variables["drawn_frame"].copy().astype(np.uint8)
            )

        cut_black_indices[selected_ind] = cut_black_indices[-1]
        cut_black_indices = cut_black_indices[:-1]

        euc_arr = euc_dist(cut_black_indices, selected_ind_val)
        selected_ind = np.argmin(euc_arr)

        counter += 1
        frame_accumulator += frame_ratio
        n_frames = int(frame_accumulator) - frames_written
        if n_frames > 0:
            frame = drawn_frame_with_hand.astype(np.uint8)
            for _ in range(n_frames):
                variables["video_object"].write(frame)
            frames_written += n_frames

        if counter % 100 == 0:
            pct = int(counter / actual_cells * 100)
            print(f"  上色进度: {pct}%")

    print(f"  上色完成，共 {counter} 步, 写入 {frames_written} 帧")


def ffmpeg_convert(source_vid, dest_vid):
    try:
        import av
        input_container = av.open(source_vid, mode="r")
        output_container = av.open(dest_vid, mode="w")
        in_stream = input_container.streams.video[0]
        width = in_stream.codec_context.width
        height = in_stream.codec_context.height
        fps = in_stream.average_rate
        out_stream = output_container.add_stream("h264", rate=fps)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "20"}
        for frame in input_container.decode(video=0):
            packet = out_stream.encode(frame)
            if packet:
                output_container.mux(packet)
        packet = out_stream.encode(None)
        if packet:
            output_container.mux(packet)
        output_container.close()
        input_container.close()
        print(f"  H.264 转码完成: {dest_vid}")
        return True
    except Exception as e:
        print(f"  FFmpeg 转码失败: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="从一张彩色图片生成白板手绘动画视频"
    )
    parser.add_argument(
        "image_path",
        help="输入图片路径"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="输出目录 (默认: ./output)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"视频总时长，单位秒 (默认: {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="禁用手部覆盖"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image_path
    output_dir = args.output_dir
    duration = args.duration
    draw_hand = not args.no_hand
    skip_rate = SKIP_RATE

    print("=" * 50)
    print("白板手绘动画生成器")
    print("=" * 50)

    # 读取图片
    print(f"\n读取图片: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"错误: 无法读取图片: {image_path}")
        sys.exit(1)

    img_ht, img_wd = image_bgr.shape[0], image_bgr.shape[1]
    print(f"  原始尺寸: {img_wd}x{img_ht}")

    # 计算目标分辨率（保持原始宽高比，长边统一缩放到 1080）
    max_dim = 1080 if MAX_1080P else max(img_wd, img_ht)
    scale = max_dim / max(img_wd, img_ht)
    img_wd = int(img_wd * scale)
    img_ht = int(img_ht * scale)
    # 确保宽高为 SPLIT_LEN 的倍数（网格切分需要）且为偶数（视频编码需要）
    lcm = SPLIT_LEN if SPLIT_LEN % 2 == 0 else SPLIT_LEN * 2
    img_wd = (img_wd // lcm) * lcm
    img_ht = (img_ht // lcm) * lcm

    print(f"  目标尺寸: {img_wd}x{img_ht}")

    # 准备输出路径
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    raw_video_path = os.path.join(output_dir, f"vid_{ts}.mp4")
    h264_video_path = os.path.join(output_dir, f"vid_{ts}_h264.mp4")

    # 初始化变量
    variables = {
        "frame_rate": FRAME_RATE,
        "resize_wd": img_wd,
        "resize_ht": img_ht,
        "split_len": SPLIT_LEN,
        "end_gray_img_duration_in_sec": END_IMG_DURATION,
        "draw_hand": draw_hand,
    }

    # 预处理图片
    print("\n预处理图片...")
    variables = preprocess_image(image_bgr, variables)

    # 预处理手部素材
    if draw_hand:
        print("加载手部素材...")
        if not os.path.exists(HAND_PATH):
            print(f"错误: 手部素材不存在: {HAND_PATH}")
            sys.exit(1)
        variables = preprocess_hand_image(HAND_PATH, variables)
        print(f"  手部尺寸: {variables['hand_wd']}x{variables['hand_ht']}")

    # 根据 duration 反算每阶段目标格子数
    end_frames = FRAME_RATE * END_IMG_DURATION
    total_frames = duration * FRAME_RATE
    anim_frames = total_frames - end_frames
    frames_per_phase = anim_frames // 2
    target_cells = frames_per_phase * skip_rate
    print(f"\n时长计算: {duration}秒 = {total_frames}帧")
    print(f"  动画帧: {anim_frames}, 每阶段: {frames_per_phase}帧")
    print(f"  skip_rate={skip_rate}, 目标格子数: {target_cells}")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    variables["video_object"] = cv2.VideoWriter(
        raw_video_path, fourcc, FRAME_RATE, (img_wd, img_ht)
    )

    # 创建空白画布
    variables["drawn_frame"] = (
        np.zeros(variables["img"].shape, np.uint8)
        + np.array([255, 255, 255], np.uint8)
    )

    # 开始绘制动画
    print(f"\n开始生成动画 (split_len={SPLIT_LEN}, skip_rate={skip_rate})...")
    start_time = time.time()

    draw_masked_object(variables, target_cells, skip_rate=skip_rate)

    # 第二阶段：上色
    print("\n开始上色阶段...")
    colorize_animation(
        variables, target_cells, skip_rate=skip_rate, brush_radius=50
    )

    # 结尾展示完整彩色原图
    end_img = variables["img"]
    for i in range(FRAME_RATE * END_IMG_DURATION):
        variables["video_object"].write(end_img)

    variables["video_object"].release()

    elapsed = time.time() - start_time
    print(f"\n原始视频生成完成: {raw_video_path}")
    print(f"  耗时: {elapsed:.1f}秒")

    # H.264 转码
    print("\n转码为 H.264...")
    if ffmpeg_convert(raw_video_path, h264_video_path):
        os.unlink(raw_video_path)
        final_path = h264_video_path
    else:
        final_path = raw_video_path

    # 获取文件大小
    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"\n最终视频: {final_path}")
    print(f"  文件大小: {size_mb:.1f} MB")
    print("=" * 50)
    print("完成!")


if __name__ == "__main__":
    main()
