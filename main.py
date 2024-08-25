import argparse

from data_shift import CSFilter


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyframes-dir', type=str, default='./keyframes')
    parser.add_argument('--path-result', type=str, default='./results')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_arg()
    
    csf = CSFilter(
        threshold=0.2
    )
    
    csf.filter(
        args.keyframes_dir,
        args.path_result
    )
    