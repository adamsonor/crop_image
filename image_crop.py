from PIL import Image
from PIL import ImageDraw
import cv2

Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_edges(image: np.ndarray, row_parts: int, col_parts: int, threshold: int) -> list[np.ndarray]:
    """画像から原稿の稜線を検出す

    :param image: np.ndarray(cv2.imreadで読み込んだ画像)
    :param row_parts: 行方向の分割数
    :param col_parts: 列方向の分割数
    :param threshold: 導関数の閾値
    :return: 4つの直線のリスト(左、右、上、下)
    """
    # 等分割する
    height, width = image.shape
    part_height = height // row_parts
    part_width = width // col_parts

    l_coords = []
    r_coords = []
    t_coords = []
    b_coords = []

    # 左右方向の稜線を検出
    for i in range(row_parts):
        # 現在のパートの開始行のインデックスを計算
        start_row = i * part_height

        # 行方向のデータを抜き出し導関数を計算
        row_data = image[start_row, :]
        derivative = np.gradient(row_data)

        # 導関数が最初から数えて閾値を超えた要素の要素番号を取得
        indices = np.where(derivative > threshold)[0]
        if indices.size > 0:
            l_coords.append((int(indices[0]), start_row))

        # 導関数が後ろから数えて-閾値を下回った要素の要素番号を取得
        reverse_indices = np.where(derivative[::-1] < -threshold)[0]
        indices = len(derivative) - reverse_indices - 1
        if indices.size > 0:
            r_coords.append((int(indices[0]), start_row))

    # 上下方向の稜線を検出
    for i in range(col_parts):
        # 現在のパートの開始列のインデックスを計算
        start_col = i * part_width

        # 列方向のデータを抜き出し導関数を計算
        col_data = image[:, start_col]
        derivative = np.gradient(col_data)

        # 導関数が最初から数えて閾値を超えた要素の要素番号を取得
        indices = np.where(derivative > threshold)[0]
        # for x in indices:
        if indices.size > 0:
            t_coords.append((start_col, int(indices[0])))

        # 導関数が後ろから数えて-閾値を下回った要素の要素番号を取得
        reverse_indices = np.where(derivative[::-1] < -threshold)[0]
        indices = len(derivative) - reverse_indices - 1
        if indices.size > 0:
            b_coords.append((start_col, int(indices[0])))

    # 最小二乗法で直線を求める
    x_l_coords, y_l_coords = zip(*l_coords)
    l_line = np.polyfit(x_l_coords, y_l_coords, 1)

    x_r_coords, y_r_coords = zip(*r_coords)
    r_line = np.polyfit(x_r_coords, y_r_coords, 1)

    x_t_coords, y_t_coords = zip(*t_coords)
    t_line = np.polyfit(x_t_coords, y_t_coords, 1)

    x_b_coords, y_b_coords = zip(*b_coords)
    b_line = np.polyfit(x_b_coords, y_b_coords, 1)

    lines = [l_line, r_line, t_line, b_line]

    return lines


def line_intersection(line1, line2) -> tuple:
    """直線の交点を計算する

    np.polyfitで求めた直線の係数を使って交点を計算する

    :param line1: 直線1の係数(a, b)
    :param line2: 直線2の係数(a, b)
    :return: 交点の座標
    """
    a1, b1 = line1
    a2, b2 = line2

    # 平行線の場合は交点が存在しない
    if a1 == a2:
        return None

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return int(x), int(y)


def calc_crop_area(lines: list[np.ndarray]) -> tuple[int, int, int, int]:
    """直線から切り取り領域を計算する

    :param lines: 原稿の稜線のリスト
    :return: 切り取り領域のパラメータ(x, y, w, h)
    """
    l_line, r_line, t_line, b_line = lines

    # 左右の直線と上下の直線の交点を求める
    lt = line_intersection(l_line, t_line)
    lb = line_intersection(l_line, b_line)
    rt = line_intersection(r_line, t_line)
    rb = line_intersection(r_line, b_line)

    # 交点が存在しない場合は None を返す
    if None in (lt, lb, rt, rb):
        return None

    # 交点から切り取り領域を計算
    x = min(lt[0], lb[0])
    y = min(lt[1], rt[1])
    w = max(rt[0], rb[0]) - x
    h = max(lb[1], rb[1]) - y

    return x, y, w, h


def crop_image(image, params: tuple[int, int, int, int]) -> np.ndarray:
    """画像を切り取る

    :param image: 画像データ
    :param params: パラメータ(x, y, w, h)
    :return: 切り取った画像
    """
    x, y, w, h = params
    cropped_image = image[y : y + h, x : x + w]
    return cropped_image


def save_image_by_cv2(image: np.ndarray, save_path: Path, dpi: str) -> None:
    """opencvで画像を保存する

    LZW圧縮でTIFF形式で保存する

    :param image: 画像データ
    :param save_path: 保存先のパス
    :param dpi: 解像度
    """
    compression_params = [
        cv2.IMWRITE_TIFF_XDPI,
        int(dpi),
        cv2.IMWRITE_TIFF_YDPI,
        int(dpi),
        cv2.IMWRITE_TIFF_COMPRESSION,
        cv2.IMWRITE_TIFF_COMPRESSION_LZW,
    ]

    cv2.imwrite(str(save_path), image, compression_params)


if __name__ == "__main__":
    # 画像の読み込み
    image_paths = Path.glob(Path.cwd(), "*.bmp")
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 稜線の検出
        lines = extract_edges(image_gray, 10, 10, 70)

        # 切り取り領域の計算
        crop_area = calc_crop_area(lines)

        # 切り取り領域が存在する場合は画像を切り取る
        if crop_area is not None:
            cropped_image = crop_image(image, crop_area)
        else:
            print("切り取り領域が存在しません")

        # 画像の保存
        save_path = image_path.stem + "_cropped.tif"
        save_image_by_cv2(cropped_image, Path(save_path), "1200")


def image_draw(image_path: Path, lines: list, coords: list) -> None:
    [l_line, r_line, t_line, b_line] = lines
    [l_coords, r_coords, t_coords, b_coords] = coords

    origin_image = cv2.imread(image_path)
    height, width, _ = origin_image.shape
    s_l_x = 0
    s_l_y = int(l_line[0] * s_l_x + l_line[1])
    e_l_x = width
    e_l_y = int(l_line[0] * e_l_x + l_line[1])
    s_r_x = 0
    s_r_y = int(r_line[0] * s_r_x + r_line[1])
    e_r_x = width
    e_r_y = int(r_line[0] * e_r_x + r_line[1])

    s_t_x = 0
    s_t_y = int(t_line[0] * s_t_x + t_line[1])
    e_t_x = width
    e_t_y = int(t_line[0] * e_t_x + t_line[1])
    s_b_x = 0
    s_b_y = int(b_line[0] * s_b_x + b_line[1])
    e_b_x = width
    e_b_y = int(b_line[0] * e_b_x + b_line[1])

    cv2.line(origin_image, (s_l_x, s_l_y), (e_l_x, e_l_y), (0, 0, 255), 20)
    cv2.line(origin_image, (s_r_x, s_r_y), (e_r_x, e_r_y), (0, 0, 255), 20)
    cv2.line(origin_image, (s_t_x, s_t_y), (e_t_x, e_t_y), (0, 0, 255), 20)
    cv2.line(origin_image, (s_b_x, s_b_y), (e_b_x, e_b_y), (0, 0, 255), 20)

    for coord in l_coords:
        cv2.circle(origin_image, coord, 50, (0, 255, 0), -1)

    for coord in r_coords:
        cv2.circle(origin_image, coord, 50, (0, 255, 0), -1)

    for coord in t_coords:
        cv2.circle(origin_image, coord, 50, (0, 255, 0), -1)

    for coord in b_coords:
        cv2.circle(origin_image, coord, 50, (0, 255, 0), -1)
