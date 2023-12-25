from pnp import pnp
from utils.dataloader import DataLoader, Dataset
from initialization import initialize

# Constants
init_bandwidth = 4

def main():
    # select dataset
    dataset = Dataset.MALAGA
    
    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=90, stride=1, steps=90)
    print("Loading data...")

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    print("Data retrieved!")
    
    # TODO: use some sort of params variable to store the best parameters
    # for each dataset; Maybe add this to the dataloader
    
    # computes landmarks and pose for the first two frames
    if dataset == Dataset.KITTI:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
    elif dataset == Dataset.PARKING:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
    elif dataset == Dataset.MALAGA:
        initialize(states[0], states[4], K)
        init_states = [states[0], states[4]]
    elif dataset == Dataset.WOKO:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]

    plot_points_cameras(
        [init_states[0].landmarks, init_states[1].landmarks],
        [init_states[0].cam_to_world, init_states[1].cam_to_world]
    )
    
    # TODO: Implement Continuous Part: Feel free to change
    start = 0
    steps = 30
    stride = 1
    for i in range(start, steps-1):
        pnp(states[i], states[i+stride], K)

    plot_points_cameras(
        [state.landmarks for state in states[0:steps-1:stride]],
        [state.cam_to_world for state in states[0:steps-1:stride]],
        plot_points=False
    )

if __name__ == '__main__':
    from plot_points_cameras import plot_points_cameras
    main()