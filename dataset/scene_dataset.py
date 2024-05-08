import os
from torch.utils.data import Dataset

class SceneDataset(Dataset):
    def __init__(self, dataset_dir, img_size=(48, 64), scale_factor=1, scene_length=120, load_into_memory=False, no_filter=False):
        # Initialize dataset parameters
        self.img_size = img_size
        self.scale_factor = scale_factor
        self.dataset_dir = dataset_dir
        self.scene_length = scene_length
        self.load_into_memory = load_into_memory
        self.no_filter = no_filter
        # List all scene files in the directory
        self.files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.npz')]
        if load_into_memory:
            self.observations = []
            self.views = []
            files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.npz')]
            for file in files:
                data = np.load(file)
                observations = data['observations']
                views = data['views']
                # Assuming channels_last format in observations
                observations = np.moveaxis(observations, -1, 1)  # Assuming channels_last format in the saved data
                y, x = observations.shape[2], observations.shape[3]
                startx = x // 2 - self.img_size[1] // 2
                starty = y // 2 - self.img_size[0] // 2
                observations = observations[:, :, starty:starty+self.img_size[0], startx:startx+self.img_size[1]]
                # Convert numpy arrays to torch tensors
                observations_tensor = torch.from_numpy(observations).float()
                views_tensor = torch.from_numpy(views).float()

                # Resize observations if necessary
                if self.scale_factor != 1:
                    observations_tensor = F.interpolate(observations_tensor, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

                self.observations.append(observations_tensor)
                self.views.append(views_tensor)
                # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # Getting usage of virtual_memory in GB ( 4th field)
            print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    def __len__(self):
        # The length of the dataset is the number of files
        return len(self.files)
    
    def __getitem__(self, index):
        if self.load_into_memory:
            observations = self.observations[index]
            views = self.views[index]
        else:
            data = np.load(self.files[index])
            observations = data['observations']
            views = data['views']

        # Adjust the number of observations to the target length if necessary
        if (observations.shape[0] > self.scene_length) and not self.no_filter:
            selected_indices = sorted(random.sample(range(observations.shape[0]), self.scene_length))
            observations = observations[selected_indices]
            views = views[selected_indices]

        if not self.load_into_memory:
            # Move axis for pytorch compatibility and crop to img_size
            observations = np.moveaxis(observations, -1, 1)  # Assuming channels_last format in the saved data
            y, x = observations.shape[2], observations.shape[3]
            startx = x // 2 - self.img_size[1] // 2
            starty = y // 2 - self.img_size[0] // 2
            observations = observations[:, :, starty:starty+self.img_size[0], startx:startx+self.img_size[1]]

            # Convert numpy arrays to torch tensors
            observations_tensor = torch.from_numpy(observations).float()
            views_tensor = torch.from_numpy(views).float()

            # Resize observations if necessary
            if self.scale_factor != 1:
                observations_tensor = F.interpolate(observations_tensor, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

            return observations_tensor, views_tensor
        return observations, views