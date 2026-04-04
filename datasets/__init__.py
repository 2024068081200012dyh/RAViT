from .lvis_dataset import LVISDataset
from .transforms import get_train_transforms, get_val_transforms

def build_dataloader(config, split='train'):
    transform = get_train_transforms() if split == 'train' else get_val_transforms()
    dataset = LVISDataset(img_dir=config['dataset']['root'], 
                         ann_file=config['dataset'][f'{split}_ann'], 
                         transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=(split=='train'))