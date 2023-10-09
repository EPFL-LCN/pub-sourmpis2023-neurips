from utils.logger import go_back_to_hunting
from optparse import OptionParser


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--log_dir",
        type="string",
        default="./log_dir/7cd678730b9444c98246c49036e400bf34b9d09f/2023_7_21_11_27_8_brain_encoding_trial",
    )
    (pars, _) = parser.parse_args()
    go_back_to_hunting(pars.log_dir, last_best="last", lr=5e-4)
