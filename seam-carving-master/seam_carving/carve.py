import warnings
from enum import Enum
from typing import Optional, Tuple
import cv2

import numba as nb
import numpy as np
from scipy.ndimage import sobel

from tqdm import tqdm
import time

DROP_MASK_ENERGY = 1e5
KEEP_MASK_ENERGY = 1e3


class OrderMode(str, Enum):
    WIDTH_FIRST = "width-first"
    HEIGHT_FIRST = "height-first"


class EnergyMode(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


def _list_enum(enum_class) -> Tuple:
    return tuple(x.value for x in enum_class)


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a grayscale image"""
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    return (rgb @ coeffs).astype(rgb.dtype)


def _get_seam_mask(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Convert a list of seam column indices to a mask"""
    # 一辺の要素数が w の単位行列(要素がすべて１)を作成
    return np.eye(src.shape[1], dtype=bool)[seam]


def _remove_seam_mask(src: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """Remove a seam from the source image according to the given seam_mask"""
    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.broadcast_to(seam_mask[:, :, None], src.shape)
        dst = src[~seam_mask].reshape((h, w - 1, c))
    else:
        h, w = src.shape
        dst = src[~seam_mask].reshape((h, w - 1))
    return dst


def _get_energy(gray: np.ndarray) -> np.ndarray:
    """Get backward energy map from the source image"""
    assert gray.ndim == 2

    gray = gray.astype(np.float32)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


@nb.njit(nb.int32[:](nb.float32[:, :]), cache=True)
def _get_backward_seam(energy: np.ndarray) -> np.ndarray:
    """Compute the minimum vertical seam from the backward energy map"""
    h, w = energy.shape
    inf = np.array([np.inf], dtype=np.float32)
    cost = np.concatenate((inf, energy[0], inf))
    parent = np.empty((h, w), dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        choices = np.vstack((cost[:-2], cost[1:-1], cost[2:]))
        min_idx = np.argmin(choices, axis=0) + base_idx
        parent[r] = min_idx
        cost[1:-1] = cost[1:-1][min_idx] + energy[r]

    c = np.argmin(cost[1:-1])
    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam


def _get_backward_seams(
    gray: np.ndarray, num_seams: int, aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """Compute the minimum N vertical seams using backward energy"""
    h, w = gray.shape
    seams = np.zeros((h, w), dtype=bool)
    rows = np.arange(h, dtype=np.int32)
    idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))
    energy = _get_energy(gray)
    if aux_energy is not None:
        energy += aux_energy
    for _ in range(num_seams):
        seam = _get_backward_seam(energy)
        seams[rows, idx_map[rows, seam]] = True

        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)

        # Only need to re-compute the energy in the bounding box of the seam
        _, cur_w = energy.shape
        lo = max(0, np.min(seam) - 1)
        hi = min(cur_w, np.max(seam) + 1)
        pad_lo = 1 if lo > 0 else 0
        pad_hi = 1 if hi < cur_w - 1 else 0
        mid_block = gray[:, lo - pad_lo : hi + pad_hi]
        _, mid_w = mid_block.shape
        mid_energy = _get_energy(mid_block)[:, pad_lo : mid_w - pad_hi]
        if aux_energy is not None:
            mid_energy += aux_energy[:, lo:hi]
        energy = np.hstack((energy[:, :lo], mid_energy, energy[:, hi + 1 :]))

    return seams


@nb.njit(
    [
        nb.int32[:](nb.float32[:, :], nb.none),
        nb.int32[:](nb.float32[:, :], nb.float32[:, :]),
    ],
    cache=True,
)
def _get_forward_seam(gray: np.ndarray, aux_energy: Optional[np.ndarray]) -> np.ndarray:
    """Compute the minimum vertical seam using forward energy"""
    """順方向エネルギーを使って最小垂直シームを計算する"""
    h, w = gray.shape

    # 水平方向に「画像の最前列」＋「画像」＋「画像の最後列」をくっつけてる
    # 多分オリジナル画像のままだと一番端だけ動的計画法できないから両端を拡張しているんだと思う
    gray = np.hstack((gray[:, :1], gray, gray[:, -1:]))

    # 無限配列を作成してる
    inf = np.array([np.inf], dtype=np.float32)

    # 1列目のエネルギーのみ絶対値差分を取ってエネルギーを計算する。両端に無限配列をくっつけているのは謎
    dp = np.concatenate((inf, np.abs(gray[0, 2:] - gray[0, :-2]), inf))

    # 多分、もとになる初期化されていない空配列を作成している
    parent = np.empty((h, w), dtype=np.int32)

    # 注目画素の左上(0), 上(1), 右上(2)で表記される min_idx を、
    # 左端からのインデックスに変換するための配列
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    inf = np.array([np.inf], dtype=np.float32)
    
    for r in range(1, h):
        # ｒ段目の1列を左右に2ピクセルずらしたもの（左右を拡張しているので１ではなく２ピクセル）
        curr_shl = gray[r, 2:]
        curr_shr = gray[r, :-2]

        # 注目画素の上の画素が削除されたあと隣接画素がくっついたときに発生するコスト
        cost_mid = np.abs(curr_shl - curr_shr)

        # 多分マスク画像コストimage.pngの追加
        if aux_energy is not None:
            cost_mid += aux_energy[r]

        # 一段上の画素を用意
        prev_mid = gray[r - 1, 1:-1]

        # 注目画素の左上もしくは右上の画素が削除されたとき隣接画素がくっついたときに発生するコスト
        cost_left = cost_mid + np.abs(prev_mid - curr_shr)
        cost_right = cost_mid + np.abs(prev_mid - curr_shl)

        # それぞれ注目画素の上・左上・右上が削除されたときの上段を用意
        dp_mid = dp[1:-1]
        dp_left = dp[:-2]
        dp_right = dp[2:]

        # 各コストの選択肢を表示
        choices = np.vstack((
            cost_left + dp_left, 
            cost_mid + dp_mid, 
            cost_right + dp_right
        ))

        # シームとして選択される各ピクセルから、コストが最小の次のピクセルへの方向を記録
        min_idx = np.argmin(choices, axis=0)

        # シームとして選択される各ピクセルから、コストが最小の次のピクセルへのインデックスを記録
        parent[r] = min_idx + base_idx

        # numba does not support specifying axis in np.min, below loop is equivalent to:
        # `dp_mid[:] = np.min(choices, axis=0)` or `dp_mid[:] = choices[min_idx, np.arange(w)]`
        for j, i in enumerate(min_idx):
            dp_mid[j] = choices[i, j]

    c = np.argmin(dp[1:-1])
    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam


def _get_forward_seams(
    gray: np.ndarray, num_seams: int, aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """Compute minimum N vertical seams using forward energy"""
    h, w = gray.shape
    seams = np.zeros((h, w), dtype=bool)
    rows = np.arange(h, dtype=np.int32)
    idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))
    for _ in range(num_seams):
        seam = _get_forward_seam(gray, aux_energy)
        seams[rows, idx_map[rows, seam]] = True
        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)

    return seams


def _get_seams(
    gray: np.ndarray, num_seams: int, energy_mode: str, aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """Get the minimum N seams from the grayscale image"""
    gray = np.asarray(gray, dtype=np.float32)
    if energy_mode == EnergyMode.BACKWARD:
        return _get_backward_seams(gray, num_seams, aux_energy)
    elif energy_mode == EnergyMode.FORWARD:
        return _get_forward_seams(gray, num_seams, aux_energy)
    else:
        raise ValueError(
            f"expect energy_mode to be one of {_list_enum(EnergyMode)}, got {energy_mode}"
        )


# ここに追加エネルギーを加える必要あり
def _reduce_width(
    src: np.ndarray,
    delta_width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reduce the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0

    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        dst_shape: Tuple[int, ...] = (src_h, src_w - delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        dst_shape = (src_h, src_w - delta_width, src_c)

    to_keep = ~_get_seams(gray, delta_width, energy_mode, aux_energy)
    dst = src[to_keep].reshape(dst_shape)
    if aux_energy is not None:
        aux_energy = aux_energy[to_keep].reshape(dst_shape[:2])
    return dst, aux_energy


@nb.njit(
    nb.float32[:, :, :](nb.float32[:, :, :], nb.boolean[:, :], nb.int32), cache=True
)
def _insert_seams_kernel(
    src: np.ndarray, seams: np.ndarray, delta_width: int
) -> np.ndarray:
    """The numba kernel for inserting seams"""
    src_h, src_w, src_c = src.shape
    dst = np.empty((src_h, src_w + delta_width, src_c), dtype=src.dtype)
    for row in range(src_h):
        dst_col = 0
        for src_col in range(src_w):
            if seams[row, src_col]:
                left = src[row, max(src_col - 1, 0)]
                right = src[row, src_col]
                dst[row, dst_col] = (left + right) / 2
                dst_col += 1
            dst[row, dst_col] = src[row, src_col]
            dst_col += 1
    return dst


def _insert_seams(src: np.ndarray, seams: np.ndarray, delta_width: int) -> np.ndarray:
    """Insert multiple seams into the source image"""
    dst = src.astype(np.float32)
    if dst.ndim == 2:
        dst = dst[:, :, None]
    dst = _insert_seams_kernel(dst, seams, delta_width).astype(src.dtype)
    if src.ndim == 2:
        dst = dst.squeeze(-1)
    return dst


def _expand_width(
    src: np.ndarray,
    delta_width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
    step_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Expand the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0
    if not 0 < step_ratio <= 1:
        raise ValueError(f"expect `step_ratio` to be between (0,1], got {step_ratio}")

    dst = src
    while delta_width > 0:
        max_step_size = max(1, round(step_ratio * dst.shape[1]))
        step_size = min(max_step_size, delta_width)
        gray = dst if dst.ndim == 2 else _rgb2gray(dst)
        seams = _get_seams(gray, step_size, energy_mode, aux_energy)
        dst = _insert_seams(dst, seams, step_size)
        if aux_energy is not None:
            aux_energy = _insert_seams(aux_energy, seams, step_size)
        delta_width -= step_size

    return dst, aux_energy


def _resize_width(
    src: np.ndarray,
    width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
    step_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize the width of image by removing vertical seams"""
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0

    src_w = src.shape[1]
    if src_w < width:
        dst, aux_energy = _expand_width(
            src, width - src_w, energy_mode, aux_energy, step_ratio
        )
    else:
        dst, aux_energy = _reduce_width(src, src_w - width, energy_mode, aux_energy)
    return dst, aux_energy


def _transpose_image(src: np.ndarray) -> np.ndarray:
    """Transpose a source image in rgb or grayscale format"""
    if src.ndim == 3:
        dst = src.transpose((1, 0, 2))
    else:
        dst = src.T
    return dst


def _resize_height(
    src: np.ndarray,
    height: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
    step_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize the height of image by removing horizontal seams"""
    assert src.ndim in (2, 3) and height > 0
    if aux_energy is not None:
        aux_energy = aux_energy.T
    src = _transpose_image(src)
    src, aux_energy = _resize_width(src, height, energy_mode, aux_energy, step_ratio)
    src = _transpose_image(src)
    if aux_energy is not None:
        aux_energy = aux_energy.T
    return src, aux_energy


def _check_mask(mask: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Ensure the mask to be a 2D grayscale map of specific shape"""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expect mask to be a 2d binary map, got shape {mask.shape}")
    if mask.shape != shape:
        raise ValueError(
            f"expect the shape of mask to match the image, got {mask.shape} vs {shape}"
        )
    return mask


def _check_src(src: np.ndarray) -> np.ndarray:
    """Ensure the source to be RGB or grayscale"""
    src = np.asarray(src)
    if src.size == 0 or src.ndim not in (2, 3):
        raise ValueError(
            f"expect a 3d rgb image or a 2d grayscale image, got image in shape {src.shape}"
        )
    return src


def resize(
    src: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    energy_mode: str = "forward",
    order: str = "width-first",
    keep_mask: Optional[np.ndarray] = None,
    drop_mask: Optional[np.ndarray] = None,
    step_ratio: float = 0.5,
) -> np.ndarray:
    """Resize the image using the content-aware seam-carving algorithm.

    :param src: A source image in RGB or grayscale format.
    :param size: The target size in pixels, as a 2-tuple (width, height).
    :param energy_mode: Policy to compute energy for the source image. Could be
        one of ``backward`` or ``forward``. If ``backward``, compute the energy
        as the gradient at each pixel. If ``forward``, compute the energy as the
        distances between adjacent pixels after each pixel is removed.
    :param order: The order to remove horizontal and vertical seams. Could be
        one of ``width-first`` or ``height-first``. In ``width-first`` mode, we
        remove or insert all vertical seams first, then the horizontal ones,
        while ``height-first`` is the opposite.
    :param keep_mask: An optional mask where the foreground is protected from
        seam removal. If not specified, no area will be protected.
    :param drop_mask: An optional binary object mask to remove. If given, the
        object will be removed before resizing the image to the target size.
    :param step_ratio: The maximum size expansion ratio in one seam carving step.
        The image will be expanded in multiple steps if target size is too large.
    :return: A resized copy of the source image.
    """

    src = _check_src(src)

    if order not in _list_enum(OrderMode):
        raise ValueError(
            f"expect order to be one of {_list_enum(OrderMode)}, got {order}"
        )

    aux_energy = None

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, src.shape[:2])
        aux_energy = np.zeros(src.shape[:2], dtype=np.float32)
        aux_energy[keep_mask] += KEEP_MASK_ENERGY

    # remove object if `drop_mask` is given
    if drop_mask is not None:
        drop_mask = _check_mask(drop_mask, src.shape[:2])

        if aux_energy is None:
            aux_energy = np.zeros(src.shape[:2], dtype=np.float32)
        aux_energy[drop_mask] -= DROP_MASK_ENERGY

        if order == OrderMode.HEIGHT_FIRST:
            src = _transpose_image(src)
            aux_energy = aux_energy.T

        num_seams = (aux_energy < 0).sum(1).max()
        while num_seams > 0:
            src, aux_energy = _reduce_width(src, num_seams, energy_mode, aux_energy)
            num_seams = (aux_energy < 0).sum(1).max()

        if order == OrderMode.HEIGHT_FIRST:
            src = _transpose_image(src)
            aux_energy = aux_energy.T

    # resize image if `size` is given
    if size is not None:
        width, height = size
        width = round(width)
        height = round(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"expect target size to be positive, got {size}")

        if order == OrderMode.WIDTH_FIRST:
            src, aux_energy = _resize_width(
                src, width, energy_mode, aux_energy, step_ratio
            )
            src, aux_energy = _resize_height(
                src, height, energy_mode, aux_energy, step_ratio
            )
        else:
            src, aux_energy = _resize_height(
                src, height, energy_mode, aux_energy, step_ratio
            )
            src, aux_energy = _resize_width(
                src, width, energy_mode, aux_energy, step_ratio
            )

    return src


def remove_object(
    src: np.ndarray, drop_mask: np.ndarray, keep_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Remove an object on the source image.

    :param src: A source image in RGB or grayscale format.
    :param drop_mask: A binary object mask to remove.
    :param keep_mask: An optional binary object mask to be protected from
        removal. If not specified, no area is protected.
    :return: A copy of the source image where the drop_mask is removed.
    """
    warnings.warn(
        "`remove_object` is deprecated in favor of `resize(src, drop_mask=mask)`, and will be removed in the next version of seam-carving",
        DeprecationWarning,
        stacklevel=2,
    )

    src = _check_src(src)

    drop_mask = _check_mask(drop_mask, src.shape[:2])

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, src.shape[:2])

    gray = src if src.ndim == 2 else _rgb2gray(src)

    while drop_mask.any():
        energy = _get_energy(gray)
        energy[drop_mask] -= DROP_MASK_ENERGY
        if keep_mask is not None:
            energy[keep_mask] += KEEP_MASK_ENERGY
        seam = _get_backward_seam(energy)
        seam_mask = _get_seam_mask(src, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        drop_mask = _remove_seam_mask(drop_mask, seam_mask)
        src = _remove_seam_mask(src, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

    return src


# ◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆ #
# 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 #
# 　　　　　　　　　　　　　　　　　　　　　　　　ここからステレオ対応用のプログラム　　　　　　　　　　　　　　　　　　　　　　　　　 #
# 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 #
# ◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆ #

@nb.jit(cache=True, nopython=True)
def calc_pair_energy(
    gray: np.ndarray,  # すでに両端は拡張済み
    disp: np.ndarray, 
    h   : int, 
    w   : int
) :
    cost_left = np.zeros(w)  # 予めサイズを確保する
    cost_mid = np.zeros(w)
    cost_right = np.zeros(w)

    for idx in range(0,w):
        # a_idx: 注目ピクセルのx座標
        if idx + disp[h, idx] > w-1:    # 画像の幅を超えて参照する場合
            a_idx = w-1
        else:
            a_idx = idx + disp[h, idx]

        # 注目ピクセルの下段3ピクセル
        # 左下
        s1 = idx-1 + disp[h-1, idx-1]
        # 中央
        s2 = idx + disp[h-1, idx]
        # 右下
        if idx+1 > w-1:
            s3 = w-1
        else:
            s3 = idx+1 + disp[h-1, idx+1]

        mid = np.abs(gray[h, a_idx-1] - gray[h, a_idx+1])

        for n,s in enumerate([s1,s2,s3]):   
            if a_idx+1 > w-1:
                cost = np.inf
                if n == 0:
                    cost_left[idx] = cost
                elif n == 1:
                    cost_mid[idx] = cost
                elif n == 2:
                    cost_right[idx] = cost
            else:
                if n == 0:
                    cost = mid + np.abs(gray[h-1, s+1:a_idx+1] - gray[h, s:a_idx])
                    cost_left[idx] = np.sum(cost)
                elif n == 1:
                    cost_mid[idx] = mid
                elif n == 2:
                    cost = mid + np.abs(gray[h-1, a_idx:s] - gray[h, a_idx+1:s+1])
                    cost_right[idx] = np.sum(cost)

    choices = np.vstack((cost_left, cost_mid, cost_right))

    return choices

# ステレオコストの重み付け
stereo_Weight = 1

@nb.jit(cache=True, nopython=True)
def _get_stereo_forward_seam(
    gray: np.ndarray, 
    disp: np.ndarray,
    aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """Compute the minimum vertical seam using forward energy"""
    """順方向エネルギーを使って最小垂直シームを計算する"""

    global Weight

    # 入力画像の高さ，幅を取得
    h, w = gray.shape

    # 水平方向に「画像の最前列」＋「画像」＋「画像の最後列」をくっつけてる
    # 多分オリジナル画像のままだと一番端だけ動的計画法できないから両端を拡張しているんだと思う
    gray = np.hstack((gray[:, :1], gray, gray[:, -1:]))

    # 画像の外側に設定するための無限値を設定
    inf = np.array([np.inf], dtype=np.float32)

    # 1列目のエネルギーのみ絶対値差分を取ってエネルギーを計算する
    dp = np.concatenate((inf, np.abs(gray[0, 2:] - gray[0, :-2]), inf))

    # 空配列を作成
    # parent: すべての画素においてコストの最小値がある方向[0:右下,1:下,2:左下]をインデックスで指定できるようにした配列
    parent = np.empty((h, w), dtype=np.int32)

    # 各インデックス位置を格納
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):

        # 対象が含まれる行を左右にずらしたものを用意
        # 左右を1ピクセル拡張しているので2ピクセルずらしている
        curr_shl = gray[r, 2:]
        curr_shr = gray[r, :-2]

        # 対象ピクセルが削除された後，左右のピクセルがくっつくことにより発生するエネルギー
        cost_mid = np.abs(curr_shl - curr_shr)

        # マスク部分にコストを追加
        if aux_energy is not None:
            cost_mid += aux_energy[r]

        # 対象ピクセルが含まれる行の一段上の行を用意
        prev_mid = gray[r - 1, 1:-1]

        # シームを削除して左右にずれたときに発生するエネルギー
        cost_left = cost_mid + np.abs(prev_mid - curr_shr)
        cost_right = cost_mid + np.abs(prev_mid - curr_shl)

        # 最上段のエネルギー
        dp_mid = dp[1:-1]
        dp_left = dp[:-2]
        dp_right = dp[2:]

        # コストを以下の形で記録

        choices = np.vstack(
            (cost_left + dp_left, cost_mid + dp_mid, cost_right + dp_right)
        )
        print(f'choices:{choices[0:]}')

        # 右視点側のコスト計算結果を加算
        stereo_cost = calc_pair_energy(gray,disp,r,w) * stereo_Weight
        print(f'stereo_cost:{stereo_cost[0:]}')
        choices = choices + stereo_cost

        # 上からコストの最小値がある方向[0:右下,1:下,2:左下]を格納
        min_idx = np.argmin(choices, axis=0)

        # 各インデックス位置の値を加算
        parent[r] = min_idx + base_idx

        # numba does not support specifying axis in np.min, below loop is equivalent to:
        # `dp_mid[:] = np.min(choices, axis=0)` or `dp_mid[:] = choices[min_idx, np.arange(w)]`
        for j, i in enumerate(min_idx):
            dp_mid[j] = choices[i, j]

    # argmin は index を出力
    c = np.argmin(dp[1:-1])
    seam = np.empty(h, dtype=np.int32)
    pair_seam = np.empty(h, dtype=np.int32)

    for r in range(h - 1, -1, -1):

        cd = c + disp[r, c]

        seam[r] = c

        if cd < w:
            pair_seam[r] = cd
        else :
            pair_seam[r] = w-1

        c = parent[r, c]

    return seam, pair_seam


def _get_stereo_forward_seams(
    gray: np.ndarray,
    pair_gray: np.ndarray,      # 追加：　グレースケール化したペア画像(右画像の予定)
    disp: np.ndarray,           # 追加：　深度マップ 
    num_seams: int, 
    aux_energy: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    
    """Compute minimum N vertical seams using forward energy"""
    h, w = gray.shape

    # シームを格納するFalseで初期化されたリスト
    seams = np.zeros((h, w), dtype=bool)
    pair_seams = np.zeros((h, w), dtype=bool)

    # 要素が入力画像の高さだけある配列
    rows = np.arange(h, dtype=np.int32)
    pair_rows = np.arange(h, dtype=np.int32)

    # ブロードキャストとは配列をshape(h, w)で指定した形状に変換する機能
    # 要素が w の配列を作成し、高さ h まで拡張する
    idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))
    pair_idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))

    # for _ in range(): は _ の変数を使用しないときに使う
    # tqdmを使用してプログレスバーを表示
    for _ in tqdm(range(num_seams)):
        # 入力画像に対するシームを作成
        seam, pair_seam = _get_stereo_forward_seam(gray, disp, aux_energy)

        # 最終的に出力する配列に選択されたシームを格納
        seams[rows, idx_map[rows, seam]] = True
        pair_seams[pair_rows, pair_idx_map[pair_rows, pair_seam]] = True

        # 以降は内部で処理するためのシームカービング
        # 削除するシームのマスクを作成
        seam_mask = _get_seam_mask(gray, seam)
        pair_seam_mask = _get_seam_mask(pair_gray, pair_seam)

        # 画像のシーム部分を削除
        gray = _remove_seam_mask(gray, seam_mask)
        pair_gray = _remove_seam_mask(pair_gray, pair_seam_mask)

        # 画像のインデックスも同じようにシーム部分を削除
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        pair_idx_map = _remove_seam_mask(pair_idx_map, pair_seam_mask)

        # 視差マップも同じ用にシーム部分を削除？
        disp = _remove_seam_mask(disp, seam_mask)

        # マスク画像もシーム部分を削る
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)
        
        time.sleep(0.1)  # 処理にかかる時間をシミュレートするための待機時間

    return seams, pair_seams


def _get_stereo_seams(
    gray: np.ndarray, 
    pair_gray: np.ndarray,      # 追加：　グレースケール化したペア画像(右画像の予定)
    disp: np.ndarray,           # 追加：　深度マップ
    num_seams: int, 
    energy_mode: str, 
    aux_energy: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the minimum N seams from the grayscale image"""
    gray = np.asarray(gray, dtype=np.float32)
    
    # 今回はforward のみ利用する
    if energy_mode == EnergyMode.FORWARD:
        return _get_stereo_forward_seams(gray, pair_gray, disp, num_seams, aux_energy)
    else:
        raise ValueError(
            f"expect energy_mode to be one of {_list_enum(EnergyMode)}, got {energy_mode}"
        )

# 画像幅の縮小
def _reduce_stereo_width(
    src: np.ndarray,
    src_pair: np.ndarray,       # 追加：　ペア画像(右画像の予定)
    disp: np.ndarray,           # 追加：　深度マップ
    delta_width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Reduce the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0

    # 入力画像がグレースケールか判断
    if src.ndim == 2:
        gray = src
        pair_gray = src_pair
        src_h, src_w = src.shape

        # dst_shape = 幅を縮小させた後の画像サイズ
        dst_shape: Tuple[int, ...] = (src_h, src_w - delta_width)   
    else:
        gray = _rgb2gray(src)
        pair_gray = _rgb2gray(src_pair)
        src_h, src_w, src_c = src.shape

        # dst_shape = 幅を縮小させた後の画像サイズ
        dst_shape = (src_h, src_w - delta_width, src_c)

    to_keep: np.ndarray
    to_keep_pair: np.ndarray

    to_keep, to_keep_pair = _get_stereo_seams(gray, pair_gray, disp, delta_width, energy_mode, aux_energy)

    dst = src[~to_keep].reshape(dst_shape)
    dst_pair = src_pair[~to_keep_pair].reshape(dst_shape)

    if aux_energy is not None:
        aux_energy = aux_energy[~to_keep].reshape(dst_shape[:2])     # 最終的には必要ないのでペアの分はつくらない

    return dst, dst_pair, aux_energy

# 画像のリサイズ(今回は縮小のみ扱う)
def _resize_stereo_width(
    src: np.ndarray,
    src_pair: np.ndarray,       # 追加：　ペア画像(右画像の予定)
    disp: np.ndarray,           # 追加：　深度マップ（多分カラー画像
    width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
    step_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
    """Resize the width of image by removing vertical seams"""
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0

    src_w = src.shape[1]

    # 画像幅と入力された数値を比較し、結果によって縮小・拡張を判断（今回は縮小のみ考える）
    dst, dst_pair, aux_energy = _reduce_stereo_width(src, src_pair, disp, src_w - width, energy_mode, aux_energy)
    
    return dst, dst_pair, aux_energy

# 画像の高さのリサイズを行う
def _resize_stereo_height(
    src: np.ndarray,
    src_pair: np.ndarray,       # 追加：　ペア画像(右画像の予定)
    disp: np.ndarray,           # 追加：　深度マップ（多分カラー画像
    height: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
    step_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
    """Resize the height of image by removing horizontal seams"""
    assert src.ndim in (2, 3) and height > 0
    if aux_energy is not None:
        aux_energy = aux_energy.T

    src = _transpose_image(src)
    src_pair = _transpose_image(src_pair)

    src, src_pair, aux_energy = _resize_stereo_width(src, src_pair, disp, height, energy_mode, aux_energy, step_ratio)

    src = _transpose_image(src)
    src_pair = _transpose_image(src_pair)

    if aux_energy is not None:
        aux_energy = aux_energy.T
    return src, src_pair,aux_energy

# 試験的にステレオコストの重みも引数にしています
def stereo_resize(
    src: np.ndarray,
    src_pair: np.ndarray,       # 追加：　ペア画像(右画像の予定)
    disp: np.ndarray,           # 追加：　深度マップ（多分カラー画像）
    weight: Optional[int],
    size: Optional[Tuple[int, int]] = None,
    energy_mode: str = "forward",
    order: str = "width-first",
    keep_mask: Optional[np.ndarray] = None,
    step_ratio: float = 0.5,
) -> np.ndarray:
    
    """内容を考慮したシームカービングアルゴリズムを用いて，画像のサイズを変更します.
    ◆ param src :  
        RGB またはグレースケール形式の入力画像.
    ◆ param size :  
        2 タプル（幅，高さ）.
    ◆ param energy_mode :  
        入力画像に対するエネルギー計算の方針.backward`` または ``forward`` のいずれか.
        backward`` の場合，各ピクセルの勾配としてエネルギーを計算する.
        forward`` の場合，各ピクセルを削除した後の隣接ピクセル間の距離としてエネルギーを計算する.
    ◆ param order : 
        水平方向と垂直方向の継ぎ目を除去する順番.
        width-first`` あるいは ``height-first`` のいずれかを指定します。
        width-first-``モードでは、まず垂直方向の継ぎ目を削除または挿入し、次に水平方向の継ぎ目を削除します。
        一方 ``height-first`` はその逆です。
    ◆ param keep_mask  : 
        前景をシーム除去から保護するオプションのマスク。
        指定しない場合は、どの領域も保護されない。
    ◆ param drop_mask  : 
        削除するバイナリオブジェクトマスク。
        与えられた場合、画像をターゲットサイズにリサイズする前にオブジェクトが除去されます。
    ◆ param step_ratio :  
        1回のシームカービングステップにおける最大サイズ拡大率.
        ターゲットサイズが大きすぎる場合、画像は複数のステップに分割されます。
    ◆ return : 
        リサイズされた元画像のコピー。
    """
    global stereo_Weight

    stereo_Weight = weight
    print(f'\nstereo_Weight : {stereo_Weight}\n')

    src = _check_src(src)
    disp_gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)

    if order not in _list_enum(OrderMode):
        raise ValueError(
            f"expect order to be one of {_list_enum(OrderMode)}, got {order}"
        )

    aux_energy = None

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, src.shape[:2])
        aux_energy = np.zeros(src.shape[:2], dtype=np.float32)
        aux_energy[keep_mask] += KEEP_MASK_ENERGY

    # resize image if `size` is given
    if size is not None:
        width, height = size
        width = round(width)
        height = round(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"expect target size to be positive, got {size}")

        if order == OrderMode.WIDTH_FIRST:
            # ステレオ化：　_resize_stereo_width, _resize_stereo_height
            src, src_pair, aux_energy = _resize_stereo_width(
                src, src_pair, disp_gray, width, energy_mode, aux_energy, step_ratio
            )
            src, src_pair, aux_energy = _resize_stereo_height(
                src, src_pair, disp_gray, height, energy_mode, aux_energy, step_ratio
            )
        else:
            src, src_pair, aux_energy = _resize_stereo_height(
                src, src_pair, disp_gray, height, energy_mode, aux_energy, step_ratio
            )
            src, src_pair, aux_energy = _resize_stereo_width(
                src, src_pair, disp_gray, width, energy_mode, aux_energy, step_ratio
            )

    return src, src_pair