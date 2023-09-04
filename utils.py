from time import time
from datetime import timedelta
import argparse

LANG_REGEX = {
    "ko": r"[ㄱ-ㅎㅏ-ㅣ가-힣]+",
    "ja": r"[ぁ-ゔァ-ヴー々〆〤ｧ-ﾝﾞﾟ]+",
    "zh": r"[\u4e00-\u9fff]+",
}
REGEX = r"[ㄱ-ㅎㅏ-ㅣ가-힣ぁ-ゔァ-ヴー々〆〤ｧ-ﾝﾞﾟ\u4e00-\u9fff]+"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epubtxt_dir", type=str, required=False, default="../bookcurpus/epubtxt",
    )
    parser.add_argument("--batch_size", type=int, required=False, default=256)
    parser.add_argument("--tokenize_in_advance", action="store_true")
    parser.add_argument("--ckpt_path", type=str, required=False)

    args = parser.parse_args()
    return args


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))

def print_number_of_parameters(model):
    print(f"""{sum([p.numel() for p in model.parameters()]):,}""")
