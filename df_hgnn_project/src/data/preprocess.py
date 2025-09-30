
import os, argparse, yaml
from .registry import load_config, auto_discover_files, build_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=None, help='YAML config path')
    ap.add_argument('--data_dir', type=str, default=None, help='raw data directory (auto discover filenames)')
    ap.add_argument('--out_dir', type=str, required=True, help='output dataset directory')
    args = ap.parse_args()

    if args.config:
        cfg = load_config(args.config)
        files = cfg.get('files', {})
        fmt = cfg.get('format', {'hyperedges':'auto','delimiter':'auto'})
        splits = cfg.get('splits', {'train':0.6,'val':0.2,'test':0.2})
        options = cfg.get('options', {})
        build_dataset(files, fmt, splits, options, args.out_dir)
    else:
        if not args.data_dir:
            raise SystemExit("Please provide --config or --data_dir")
        files = auto_discover_files(args.data_dir)
        fmt = {'hyperedges':'auto','delimiter':'auto'}
        splits = {'train':0.6, 'val':0.2, 'test':0.2}
        options = {}
        build_dataset(files, fmt, splits, options, args.out_dir)

if __name__ == '__main__':
    main()
