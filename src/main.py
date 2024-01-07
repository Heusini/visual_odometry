import argparse
from tracking import run_tracking
from utils.dataloader import Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parking', '-p', help='the parking dataset', action='store_true')
    parser.add_argument('--kitty', '-k',  help='the kitty dataset', action='store_true')
    parser.add_argument('--malaga', '-m', help='the malaga dataset', action='store_true')
    parser.add_argument('--plot', '-o', help='plot the results', action='store_true')
    args = parser.parse_args()

    dataset = Dataset.KITTI
    if args.parking:
        dataset = Dataset.PARKING
    elif args.kitty:
        dataset = Dataset.KITTI
    elif args.malaga:
        dataset = Dataset.MALAGA

    run_tracking(dataset, args.plot)

