import numpy as np
from PIL import Image
import cv2
import seam_carving

class Pass:
    src_pass    = './images/left/sample_l.png'
    pair_pass   = './images/right/sample_r.png'
    disp_pass   = './images/disp/sample_disp_map.png'
    mask_pass   = './images/mask/mask.png'

class ImageData:
    # ImageData クラスの初期化（typeの値によって保持するデータが変わる）
    def __init__(self, type: int):
        self.type = type
        self.src = cv2.imread(Pass.src_pass)

        if type == 3 or 4 or 5 :      # type = 3,4,5 のときはステレオ機能を利用する
            print(f'mode:   stereo_mode')
            self.pair = cv2.imread(Pass.pair_pass)
            self.disp = cv2.imread(Pass.disp_pass)
        
        if type == 1 or 2 or 4 or 5 :      # type = 3,4,5 のときはステレオ機能を利用する
            print(f'mask:   use_mask')
            self.mask = cv2.imread(Pass.mask_pass, cv2.IMREAD_GRAYSCALE)

        print(f'\ntype:   {type}')

    # ステレオコストの重みと縮小幅を入力する
    def set_param(self):
        self.weight     = int(input(f'ステレオコストの重みを数値で入力してください  ：'))
        self.r_width    = int(input(f'縮小したい幅を数値で入力してください          ：'))
        name            =     input(f'追加したい文字があれば入力してください        ：')
        self.save_name = f'{name}_w_{self.weight}_r_{self.r_width}'

    # 画像サイズを統一する
    def initialize(self, size: list):
        self.src    = cv2.resize(self.src,  size)
        if type == 3 or 4 or 5 :    # type = 3,4,5 のときはステレオ機能を利用する
            self.pair   = cv2.resize(self.pair, size)
            self.disp   = cv2.resize(self.disp, size)
        if type == 1 or 2 or 4 or 5 :   # type = 1,2,4,5 のときはマスクを利用する
            self.mask = cv2.resize(self.mask, size)


# 画像の処理方法を選択      type = 0　：リサイズ          　マスク:無
# type: 0 ~ 5               type = 1　：リサイズ          　マスク:有
#                           type = 2　：オブジェクト除去  　マスク:有
#                           type = 3　：ステレオリサイズ  　マスク:無
#                           type = 4　：ステレオリサイズ  　マスク:有　　※視差マップのみ
#                           type = 5　：ステレオリサイズ  　マスク:有　　※ステレオエネルギーあり
type = 5

# インスタンスの生成
img = ImageData(type)
img.set_param()

# 画像サイズの取得
h , w , c = img.src.shape   # src  のサイズを取得
size = (w,h)

# 画像の正規化
img.initialize(size)

# 入力画像の情報を表示
print(f'\n')
print(f'src(h,w)        : {img.src.shape[:2]}')
print(f'pair(h,w)       : {img.pair.shape[:2]}')
print(f'disp(h,w)       : {img.disp.shape[:2]}')

if type == 0:
    # 画像幅(w)から img.r_width だけ増減させる
    scale_down = seam_carving.resize(img.src, (w - img.img.r_width, h))
    scale_up = seam_carving.resize(img.src, (w + img._width, h))

    # 画像をつなぐ際の空白スペース
    padding = np.full((h, 8, c), 255, dtype=np.uint8)

    # PILの機能で結果を表示
    show_img = Image.fromarray(np.hstack((img.src, padding, scale_down, padding, scale_up)))
    show_img.show()

elif type == 1:
    # 画像幅(w)から img.r_width だけ増減させる
    scale_down = seam_carving.resize(img.src, (w - img.r_width, h), energy_mode = 'backward', keep_mask = img.mask)
    scale_up = seam_carving.resize(img.src, (w + img.r_width, h), keep_mask = img.mask)

    # 画像をつなぐ際の空白スペース
    padding = np.full((h, 8, c), 255, dtype=np.uint8)

    # PILの機能で結果を表示
    show_img = Image.fromarray(np.hstack((img.src, padding, scale_down, padding, scale_up)))

elif type == 2:
    # 物体除去
    remove_mask = seam_carving.resize(img.src, drop_mask = img.mask)

    # 画像をつなぐ際の空白スペース
    padding = np.full((h, 8, c), 255, dtype=np.uint8)

    # PILの機能で結果を表示
    show_img = Image.fromarray(np.hstack((img.src, padding, remove_mask)))

elif type == 3:
    # 画像幅(w)から img.r_width だけ増減させる
    # scale_down = seam_carving.resize(src, (w - img.r_width, h))
    scale_down, scale_down_pair = seam_carving.stereo_resize(img.src, img.pair, img.disp, (w - img.r_width, h))

    # 画像をつなぐ際の空白スペース
    padding = np.full((h, 8, c), 255, dtype=np.uint8)

    # 画像の保存
    cv2.imwrite(f'./results/no_mask/{img.save_name}_l.png', scale_down)
    cv2.imwrite(f'./results/no_mask/{img.save_name}_r.png', scale_down_pair)

    # PIL は RGB 前提, imread は BGR で読み込むため RGB 変換が必要
    scale_down      = cv2.cvtColor(scale_down      ,cv2.COLOR_BGR2RGB) 
    scale_down_pair = cv2.cvtColor(scale_down_pair ,cv2.COLOR_BGR2RGB)

    # PILの機能で結果を表示
    # show_img = Image.fromarray(np.hstack((img.disp, padding, scale_down, padding, scale_down_pair)))
    show_img = Image.fromarray(np.hstack((scale_down, padding, scale_down_pair)))

elif type == 4:
    # 画像幅(w)から img.r_width だけ増減させる
    scale_down, scale_down_pair = seam_carving.stereo_resize(img.src, img.pair, img.disp, img.weight, (w - img.r_width, h), keep_mask = img.mask)

    # 画像をつなぐ際の空白スペース
    padding = np.full((h, 8, c), 255, dtype=np.uint8)

    # 画像の保存
    cv2.imwrite(f'./results/mask/only_disp/left/{img.save_name}.png', scale_down)
    cv2.imwrite(f'./results/mask/only_disp/right/{img.save_name}.png', scale_down_pair)

    # PIL は RGB 前提, imread は BGR で読み込むため RGB 変換が必要
    scale_down      = cv2.cvtColor(scale_down      ,cv2.COLOR_BGR2RGB)
    scale_down_pair = cv2.cvtColor(scale_down_pair ,cv2.COLOR_BGR2RGB)

    # PILの機能で結果を表示
    show_img = Image.fromarray(np.hstack((img.disp, padding, scale_down, padding, scale_down_pair)))

    show_img.save(f'./results/mask/only_disp/compare/{img.save_name}_compare_img.png')

elif type == 5:
    # 画像幅(w)から img.r_width だけ増減させる
    scale_down, scale_down_pair = seam_carving.stereo_resize(img.src, img.pair, img.disp, img.weight, (w - img.r_width, h), keep_mask = img.mask)

    print(f'\n')
    print(f'result      :{scale_down.shape[:2]}')
    print(f'result_pair :{scale_down_pair.shape[:2]}')

    # 画像をつなぐ際の空白スペース
    padding = np.full((h, 8, c), 255, dtype=np.uint8)

    # 画像の保存
    cv2.imwrite(f'./results/mask/stereo_energy/left/{img.save_name}.png', scale_down)
    cv2.imwrite(f'./results/mask/stereo_energy/right/{img.save_name}.png', scale_down_pair)

    # PIL は RGB 前提, imread は BGR で読み込むため RGB 変換が必要
    scale_down      = cv2.cvtColor(scale_down      ,cv2.COLOR_BGR2RGB)
    scale_down_pair = cv2.cvtColor(scale_down_pair ,cv2.COLOR_BGR2RGB)

    # PILの機能で結果を表示
    # show_img = Image.fromarray(np.hstack((img.disp, padding, scale_down, padding, scale_down_pair)))
    show_img = Image.fromarray(np.hstack((scale_down, padding, scale_down_pair)))
    SBS_img = Image.fromarray(np.hstack((scale_down, scale_down_pair)))
    show_img.save(f'./results/mask/stereo_energy/compare/{img.save_name}_compare_img.png')
    SBS_img.save(f'./results/mask/stereo_energy/SBS/{img.save_name}.png')


show_img.show()