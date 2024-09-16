from PIL import Image
from PIL import ImageDraw
import cv2

Image.MAX_IMAGE_PIXELS = None


# A3サイズの画像作成
def create_a3_image(dpi=1200):
    width_mm = 420
    height_mm = 297

    # mmをピクセルに変換
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4

    # ピクセル数を計算
    a3_w = int(width_in * dpi)
    a3_h = int(height_in * dpi)

    a3_image = Image.new("RGB", (a3_w, a3_h), "black")

    return a3_image


def create_grid_image(image_size=(215.5, 279.4), dpi=1200, grid_size_mm=10, margin_mm=None, rotate=None):
    # 画像サイズ（単位：mm）
    width_mm, height_mm = image_size

    # mmをピクセルに変換
    pixels_per_mm = dpi / 25.4
    width_px = int(width_mm * pixels_per_mm)
    height_px = int(height_mm * pixels_per_mm)
    grid_size_px = int(grid_size_mm * pixels_per_mm)

    margin_px = 0
    if margin_mm is not None:
        margin_px = int(margin_mm * pixels_per_mm)

    # 白色の背景で新しい画像を作成
    image = Image.new("RGB", (width_px, height_px), color=(220, 220, 220))
    draw = ImageDraw.Draw(image)

    # マージンを考慮して格子線を描画
    for x in range(margin_px, width_px - margin_px, grid_size_px):
        draw.line([(x, margin_px), (x, height_px - margin_px)], fill=(50, 50, 50), width=5)
    for y in range(margin_px, height_px - margin_px, grid_size_px):
        draw.line([(margin_px, y), (width_px - margin_px, y)], fill=(50, 50, 50), width=5)

    # 回転を考慮
    if rotate:
        image = image.rotate(rotate, expand=True)

    return image


if __name__ == "__main__":
    # A3サイズの画像を作成
    a3_image = create_a3_image()

    # グリッド線の画像を作成
    rotate = -1
    grid_size_mm = 10
    grid_image = create_grid_image(
        image_size=(215.5, 279.4), dpi=1200, grid_size_mm=grid_size_mm, margin_mm=None, rotate=rotate
    )

    # A3サイズの画像にグリッド線画像を貼り付け
    a3_image.paste(grid_image, (100, 100))

    # 画像を保存
    save_path = r"sample\a3_image" + f"_g{grid_size_mm}_r{rotate}" + ".bmp"
    a3_image.save(save_path, format="BMP")

    # # 画像を表示
    # cv2.imshow("A3 Image", cv2.imread("a3_image.png"))
    # cv2.imshow("Grid Image", cv2.imread("grid_image.png"))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
