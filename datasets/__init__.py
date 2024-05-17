from .sceneflow_dataset import SceneFlowDatset
from .kitti_dataset import KITTIDataset




__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
}
