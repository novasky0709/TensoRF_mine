from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .qingdao_dataset import QingDaoDataset


dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
               'qdd':QingDaoDataset,
               'own_data':YourOwnDataset}