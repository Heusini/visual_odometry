from twoDtwoD import twoDtwoD
from utils.dataloader import DataLoader, Dataset
from initialization import initialize
from pnp import pnp

# Constants
init_bandwidth = 4

def main():
    # select dataset
    dataset = Dataset.KITTI
    
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
    steps = 0
    while steps < len(states) - 3:
        if steps == 90:
            break
        
        # if only 20 % of the landmarks are visible, call initialize again
        # currently it is not adding the keypoints and landmarks but rather replacing the current landmarks and kp
        # this step should be then removed with KLT where we add new keypoints at each frame
        # (probably a good idea is to store this in a separate variable in framestate, since 
        # this newly added keypoints can only be used at the point where new landmarks are created
        # see project statement last page)
        # TODO: Implement 4.3 (see project statement)
        if states[steps].landmarks.shape[1] / states[0].landmarks.shape[1] < 0.2:
            initialize(states[steps], states[steps + 3], K)

        pnp(states[steps], states[steps+1], K)
        steps += 1

    plot_points_cameras(
        [state.landmarks for state in states[0:steps-1]],
        [state.cam_to_world for state in states[0:steps-1]],
        plot_points=False
    )

if __name__ == '__main__':
    from plot_points_cameras import plot_points_cameras
    main()