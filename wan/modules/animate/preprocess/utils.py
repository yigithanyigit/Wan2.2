# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import cv2
import math
import random
import numpy as np

def get_mask_boxes(mask):
    """

    Args:
        mask: [h, w]
    Returns:

    """
    y_coords, x_coords = np.nonzero(mask)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(np.int32)
    return bbox


def get_aug_mask(body_mask, w_len=10, h_len=20):
    if not body_mask.any():
        return body_mask
    
    body_bbox = get_mask_boxes(body_mask)

    bbox_wh = body_bbox[2:4] - body_bbox[0:2]
    w_slice = np.int32(bbox_wh[0] / w_len)
    h_slice = np.int32(bbox_wh[1] / h_len)

    for each_w in range(body_bbox[0], body_bbox[2], w_slice):
        w_start = min(each_w, body_bbox[2])
        w_end = min((each_w + w_slice), body_bbox[2])
        # print(w_start, w_end)
        for each_h in range(body_bbox[1], body_bbox[3], h_slice):
            h_start = min(each_h, body_bbox[3])
            h_end = min((each_h + h_slice), body_bbox[3])
            if body_mask[h_start:h_end, w_start:w_end].sum() > 0:
                body_mask[h_start:h_end, w_start:w_end] = 1

    return body_mask
    
def get_mask_body_img(img_copy, hand_mask, k=7, iterations=1):
    kernel = np.ones((k, k), np.uint8)
    dilation = cv2.dilate(hand_mask, kernel, iterations=iterations)
    mask_hand_img = img_copy * (1 - dilation[:, :, None])

    return mask_hand_img, dilation


def get_face_bboxes(kp2ds, scale, image_shape, ratio_aug):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[23:91, :2]

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)


    initial_width = max_x - min_x
    initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    if ratio_aug:
        if random.random() > 0.5:
            delta_width += random.uniform(0, initial_width // 10)
        else:
            delta_height += random.uniform(0, initial_height // 10)

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]


def calculate_new_size(orig_w, orig_h, target_area, divisor=64):

    target_ratio = orig_w / orig_h

    def check_valid(w, h):

        if w <= 0 or h <= 0:
            return False
        return (w * h <= target_area and  
                w % divisor == 0 and  
                h % divisor == 0)  

    def get_ratio_diff(w, h):

        return abs(w / h - target_ratio)

    def round_to_64(value, round_up=False, divisor=64):

        if round_up:
            return divisor * ((value + (divisor - 1)) // divisor)
        return divisor * (value // divisor)

    possible_sizes = []

    max_area_h = int(np.sqrt(target_area / target_ratio))
    max_area_w = int(max_area_h * target_ratio)

    max_h = round_to_64(max_area_h, round_up=True, divisor=divisor)
    max_w = round_to_64(max_area_w, round_up=True, divisor=divisor)

    for h in range(divisor, max_h + divisor, divisor):
        ideal_w = h * target_ratio

        w_down = round_to_64(ideal_w)
        w_up = round_to_64(ideal_w, round_up=True)

        for w in [w_down, w_up]:
            if check_valid(w, h, divisor):
                possible_sizes.append((w, h, get_ratio_diff(w, h)))

    if not possible_sizes:
        raise ValueError("Can not find suitable size")

    possible_sizes.sort(key=lambda x: (-x[0] * x[1], x[2]))

    best_w, best_h, _ = possible_sizes[0]
    return int(best_w), int(best_h)


def resize_by_area(image, target_area, keep_aspect_ratio=True, divisor=64, padding_color=(0, 0, 0)):
    h, w = image.shape[:2]
    try:
        new_w, new_h = calculate_new_size(w, h, target_area, divisor)
    except:
        aspect_ratio = w / h

        if keep_aspect_ratio:
            new_h = math.sqrt(target_area / aspect_ratio)
            new_w = target_area / new_h
        else:
            new_w = new_h = math.sqrt(target_area)

        new_w, new_h = int((new_w // divisor) * divisor), int((new_h // divisor) * divisor)

    interpolation = cv2.INTER_AREA if (new_w * new_h < w * h) else cv2.INTER_LINEAR

    resized_image = padding_resize(image, height=new_h, width=new_w, padding_color=padding_color,
                                    interpolation=interpolation)
    return resized_image


def padding_resize(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    ori_height = img_ori.shape[0]
    ori_width = img_ori.shape[1]
    channel = img_ori.shape[2]

    img_pad = np.zeros((height, width, channel))
    if channel == 1:
        img_pad[:, :, 0] = padding_color[0]
    else:
        img_pad[:, :, 0] = padding_color[0]
        img_pad[:, :, 1] = padding_color[1]
        img_pad[:, :, 2] = padding_color[2]

    if (ori_height / ori_width) > (height / width):
        new_width = int(height / ori_height * ori_width)
        img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
        padding = int((width - new_width) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]  
        img_pad[:, padding: padding + new_width, :] = img
    else:
        new_height = int(width / ori_width * ori_height)
        img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
        padding = int((height - new_height) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]  
        img_pad[padding: padding + new_height, :, :] = img

    img_pad = np.uint8(img_pad)

    return img_pad


def get_frame_indices(frame_num, video_fps, clip_length, train_fps):

    start_frame = 0
    times = np.arange(0, clip_length) / train_fps
    frame_indices = start_frame + np.round(times * video_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, frame_num - 1)

    return frame_indices.tolist()


def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp = np.asarray(kp2ds, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] < 2:
        return [0, w, 0, h]

    pts = kp.copy()[:, :2]
    pts = pts[1:] if pts.shape[0] > 1 else pts
    pts = pts * (w, h)

    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if pts.size == 0:
        return [0, w, 0, h]

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    if not np.isfinite(initial_width) or not np.isfinite(initial_height) or initial_width <= 0 or initial_height <= 0:
        cx = float(np.clip(np.nanmean(pts[:, 0]), 0, max(w - 1, 0)))
        cy = float(np.clip(np.nanmean(pts[:, 1]), 0, max(h - 1, 0)))
        side = 0.1 * min(w, h) * np.sqrt(max(scale, 1e-6))
        half = side / 2.0
        x1 = max(0.0, cx - half)
        x2 = min(float(w), cx + half)
        y1 = max(0.0, cy - half)
        y2 = min(float(h), cy + half)
        return [int(x1), int(x2), int(y1), int(y2)]

    # original computation path (unchanged semantics for valid inputs)
    initial_area = initial_width * initial_height
    expanded_area = initial_area * scale
    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2.0
    delta_height = (new_height - initial_height) / 4.0

    expanded_min_x = max(min_x - delta_width, 0.0)
    expanded_max_x = min(max_x + delta_width, float(w))
    expanded_min_y = max(min_y - 3.0 * delta_height, 0.0)
    expanded_max_y = min(max_y + delta_height, float(h))

    if not np.isfinite(expanded_min_x):
        expanded_min_x = 0.0
    if not np.isfinite(expanded_max_x):
        expanded_max_x = float(w)
    if not np.isfinite(expanded_min_y):
        expanded_min_y = 0.0
    if not np.isfinite(expanded_max_y):
        expanded_max_y = float(h)

    if expanded_max_x <= expanded_min_x or expanded_max_y <= expanded_min_y:
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        side = max(initial_width, initial_height, 1.0)
        half = side / 2.0
        expanded_min_x = max(0.0, cx - half)
        expanded_max_x = min(float(w), cx + half)
        expanded_min_y = max(0.0, cy - half)
        expanded_max_y = min(float(h), cy + half)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]