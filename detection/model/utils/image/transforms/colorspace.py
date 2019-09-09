import cv2


def convert_color_factory(src, dst):
    code = getattr(cv2, 'COLOR_{}2{}'.format(src.upper(), dst.upper()))

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = """Convert a {0} image to {1} image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted {1} image.
    """.format(src.upper(), dst.upper())

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')
rgb2bgr = convert_color_factory('rgb', 'bgr')