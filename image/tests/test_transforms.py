import numpy as np

from ai4eyes_image.preprocess.transforms import crop, render_train, prepare, square_crop_box, to_rgb_448, token_keep


def test_crop_zero_pads_outside_the_frame():
    img = np.ones((10, 10), np.uint8)
    c = crop(img, (-5, -5, 10))
    assert c.shape == (10, 10) and c[:5].sum() == 0 and c[5:, 5:].all()


def test_square_box_and_token_keep():
    m = np.zeros((1080, 1920), bool)
    m[300:700, 800:1200] = True
    x0, y0, side = square_crop_box(m)
    assert side == round(400 * 1.08)
    cf = np.zeros((448, 448), np.float32)
    cf[:224] = 1
    rf = np.zeros_like(cf)
    k = token_keep(cf, rf)
    assert k.shape == (28 * 28,) and k.sum() == 14 * 28
    rf[0, 0] = 1                                                       # any ring pixel drops that token
    assert token_keep(cf, rf).sum() == 14 * 28 - 1


def test_eval_and_train_views_have_the_same_shape():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    frame, none = np.ones((540, 960), bool), np.zeros((540, 960), bool)
    x, cf, rf = to_rgb_448(img, frame, none)
    xt, cft, rft = render_train(prepare(img, frame, none), rng)
    assert x.shape == xt.shape == (3, 448, 448) and cf.shape == cft.shape == (448, 448)
