from PIL import Image
from PIL import ImageDraw
import cv2

Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from scipy import stats

import numpy as np
from scipy import stats

import time


def check_dpi(image: np.ndarray) -> int:
    """画像の解像度取得

    A3サイズ領域に対して何dotあるかで解像度を計算

    :param image: 画像データ
    :return: 解像度
    """
    height, width = image.shape
    a3_height = 297
    a3_width = 420
    inch_h = a3_height / 25.4
    inch_w = a3_width / 25.4
    dpi_h = int(round(height // inch_h, -2))
    dpi_w = int(round(width // inch_w, -2))

    return (dpi_h, dpi_w)


def remove_outliers_linear(data, threshold=1.5) -> list[tuple[int, int]] | None:
    """外れ値を除去する

    :param data: 座標(x,y)のリスト
    :param threshold: Zスコアの閾値, defaults to 1.5
    :return: 外れ値を除外した座標のリスト or None
    """
    X = np.array([point[0] for point in data]).reshape(-1, 1)
    y = np.array([point[1] for point in data])

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    # 標準偏差がゼロの場合(直線上にすべて乗っている場合)、すべてを返す
    if np.std(residuals) == 0:
        return data

    z_scores = np.abs(stats.zscore(residuals))

    return [data[i] for i in range(len(data)) if z_scores[i] <= threshold]


def extract_edges(
    image: np.ndarray, row_parts: int, col_parts: int, threshold: int
) -> tuple[list[np.ndarray] | list[int], list[np.ndarray]]:
    """画像から原稿の稜線を検出す

    :param image: np.ndarray(cv2.imreadで読み込んだ画像)
    :param row_parts: 行方向の分割数
    :param col_parts: 列方向の分割数
    :param threshold: 導関数の閾値
    :return: 4つの直線のリスト(左、右、上、下), 4つの直線上の座標のリスト
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

        # 導関数の後ろから数えて-閾値を下回った要素の要素番号を取得
        reverse_indices = np.where(derivative[::-1] < -threshold)[0]
        indices = len(derivative) - reverse_indices - 1
        if indices.size > 0:
            b_coords.append((start_col, int(indices[0])))

    coords = [l_coords, r_coords, t_coords, b_coords]

    lines = []
    # 各方向の直線を求める
    for i, each_coords in enumerate(coords):
        # 外れ値を除去
        removed_coords = remove_outliers_linear(each_coords)
        x_coords, y_coords = zip(*removed_coords)
        # x座標がすべて同じ場合はx座標を追加
        if np.unique(x_coords).size == 1:
            lines.append(x_coords[0])
        # y座標がすべて同じ場合は傾き0、切片にy座標を追加
        elif np.unique(y_coords).size == 1:
            lines.append(np.array([0, y_coords[0]]))
        else:
            # 最小二乗法で直線を求める
            line = np.polyfit(x_coords, y_coords, 1)
            lines.append(line)

    # l_coords = remove_outliers_linear(l_coords)
    # r_coords = remove_outliers_linear(r_coords)
    # t_coords = remove_outliers_linear(t_coords)
    # b_coords = remove_outliers_linear(b_coords)

    # # 最小二乗法で直線を求める
    # x_l_coords, y_l_coords = zip(*l_coords)
    # l_line = np.polyfit(x_l_coords, y_l_coords, 1)

    # x_r_coords, y_r_coords = zip(*r_coords)
    # r_line = np.polyfit(x_r_coords, y_r_coords, 1)

    # x_t_coords, y_t_coords = zip(*t_coords)
    # t_line = np.polyfit(x_t_coords, y_t_coords, 1)

    # x_b_coords, y_b_coords = zip(*b_coords)
    # b_line = np.polyfit(x_b_coords, y_b_coords, 1)

    # lines = [l_line, r_line, t_line, b_line]
    # coords = [l_coords, r_coords, t_coords, b_coords]

    return lines, coords


def calc_angle(lines: list[np.ndarray | int]) -> float:
    """直線の傾きを求める

    :param line: 直線の係数(a, b)
    :return: 傾き(deg)
    """
    (
        l_line,
        _,
        _,
        _,
    ) = lines

    # l_lineがintの場合(x=〇〇の直線)は0を返す
    if type(l_line) == int:
        return 0

    l_angle = np.rad2deg(np.arctan(l_line[0]))

    return l_angle


def line_intersection(line1: int | np.ndarray, line2: int | np.ndarray) -> tuple[int, int] | None:
    """直線の交点を計算する

    np.polyfitで求めた直線の係数を使って交点を計算する

    :param line1: 直線1の係数(a, b)
    :param line2: 直線2の係数(a, b)
    :return: 交点の座標
    """
    if type(line1) == int:
        x = line1
        y = line2[0] * x + line2[1]
    elif type(line2) == int:
        x = line2
        y = line1[0] * x + line1[1]
    else:
        a1, b1 = line1
        a2, b2 = line2
        # 平行線の場合は交点が存在しない
        if a1 == a2:
            return None
        x = (b2 - b1) / (a1 - a2)
        y = a1 * x + b1

    return int(x), int(y)


def calc_crop_area(lines: list[np.ndarray | int], image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """直線から切り取り領域を計算する

    交点が無いもしくは画像領域外の場合は None を返す

    :param lines: 原稿の稜線のリスト
    :param image_shape: 画像のサイズ(height, width)
    :return: 切り取り領域のパラメータ(x, y, w, h)
    """
    l_line, r_line, t_line, b_line = lines

    # 左右の直線と上下の直線の交点を求める
    lt = line_intersection(l_line, t_line)
    lb = line_intersection(l_line, b_line)
    rt = line_intersection(r_line, t_line)
    rb = line_intersection(r_line, b_line)

    # 交点が存在しない、もしくは画像領域外の場合は None を返す
    height, width = image_shape
    judge = [0 <= coord[0] < width and 0 <= coord[1] < height for coord in (lt, lb, rt, rb)]
    if None in (lt, lb, rt, rb) or not all(judge):
        return None

    # 交点から切り取り領域を計算
    x = min(lt[0], lb[0])
    y = min(lt[1], rt[1])
    w = max(rt[0], rb[0]) - x
    h = max(lb[1], rb[1]) - y

    return x, y, w, h


def crop_image(image, params: tuple[int, int, int, int]) -> np.ndarray:
    """画像を指定領域で切り取る

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
    ]

    cv2.imwrite(str(save_path), image, compression_params)


def save_process_image(image: np.ndarray, lines: list[np.ndarray | int], coords: list) -> None:
    """処理結果を描画して保存する

    :param image: 画像データ
    :param lines: 直線のリスト
    :param coords: 座標のリスト
    """
    # 描画用の色設定
    colors = [(100, 100, 255), (255, 100, 100), (100, 255, 100), (150, 150, 150)]

    _, width, _ = image.shape
    for i, line in enumerate(lines):
        if type(line) == int:
            cv2.line(image, (line, 0), (line, width), colors[i], 20)
        else:
            poly_line = np.poly1d(line)

            # 直線の端点を計算
            x_min, x_max = 0, width
            y_min, y_max = int(poly_line(x_min)), int(poly_line(x_max))

            # 直線を描画
            cv2.line(image, (x_min, y_min), (x_max, y_max), colors[i], 20)

    for i, each_coord in enumerate(coords):
        for coord in each_coord:
            cv2.circle(image, coord, 50, colors[i], -1)

    save_path = image_path.stem + "_drawn.tif"
    cv2.imwrite(save_path, image)


if __name__ == "__main__":
    # 画像の読み込み
    # image_paths = Path.glob(Path.cwd() / Path("sample"), "*.bmp")
    image_paths = [Path(r"sample\a3_image_g10_r1.bmp")]
    for image_path in image_paths:
        start_time = time.time()
        image = cv2.imread(str(image_path))
        end_time = time.time()
        print(f"画像読み込み時間: {end_time - start_time:.4f}秒")

        # 縦長画像の場合に左に90°回転
        start_time = time.time()
        height, width, _ = image.shape
        if height > width:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        end_time = time.time()
        print(f"画像回転時間: {end_time - start_time:.4f}秒")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 解像度の情報取得
        dpi = check_dpi(image_gray)

        # 稜線の検出
        start_time = time.time()
        lines, coords = extract_edges(image_gray, 10, 10, 70)
        end_time = time.time()
        print(f"稜線検出時間: {end_time - start_time:.4f}秒")

        # 傾きを求める
        start_time = time.time()
        angle = calc_angle(lines)
        end_time = time.time()
        print(f"傾き計算時間: {end_time - start_time:.4f}秒")

        # 切り取り領域の計算
        start_time = time.time()
        crop_area = calc_crop_area(lines, image_gray.shape)
        if crop_area is None:
            crop_area = (0, 0, width, height)
        end_time = time.time()
        print(f"切り取り領域計算時間: {end_time - start_time:.4f}秒")

        # 切り取り領域が存在する場合は画像を切り取る
        start_time = time.time()
        if crop_area is not None:
            cropped_image = crop_image(image, crop_area)
        else:
            print("切り取り領域が存在しません")
        end_time = time.time()
        print(f"画像切り取り時間: {end_time - start_time:.4f}秒")

        # 画像の保存
        start_time = time.time()
        save_path = image_path.stem + "_cropped.tif"
        image_pil = Image.fromarray(cropped_image)
        image_pil.save(save_path, dpi=dpi, format="TIFF")
        end_time = time.time()
        print(f"画像保存時間: {end_time - start_time:.4f}秒")

        start_time = time.time()
        save_process_image(image, lines, coords)
        end_time = time.time()
        print(f"処理結果保存時間: {end_time - start_time:.4f}秒")
