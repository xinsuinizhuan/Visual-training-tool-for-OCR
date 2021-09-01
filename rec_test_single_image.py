import argparse
import time
import cv2
from tools import RecInfer


def main():
    parser = argparse.ArgumentParser(description='rec net')
    parser.add_argument('--model_path', default='save_model/crnn_epoch0_word_acc0.962000_char_acc0.990500.pth',
                        help='model path')  
    parser.add_argument('--model_type', default='crnn',
                        help='model_type', type=str)
    parser.add_argument('--nh', default=256, type=int, help='nh')
    parser.add_argument('--img_H', default=32, type=int, help='img_H')
    parser.add_argument('--use_lstm', default=False,
                        help='use_lstm', type=bool)
    cfg = parser.parse_args()
    infer = RecInfer(cfg)

    img = cv2.imread('test1/00000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    out = infer.predict(img)
    t1 = time.time()
    print(out, (t1-t0)*1000)


if __name__ == "__main__":
    main()
