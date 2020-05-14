import enum

class OrderedEnum(enum.Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
class Run_types(OrderedEnum):
    all_heat_paf_l2_loss = 0
    all_heat_paf_l2_reg_loss = 1
    all_heat_paf_kl_loss = 2
    
    
class Run_settings():
    def __init__(self, run_type=Run_types.all_heat_paf_l2_loss):
        self.run_type = run_type
        
    def get_settings(self):
        snapshot_name = ''
        train_dir = ''
        test_dir = ''
        if self.run_type == Run_types.all_heat_paf_l2_loss:
            snapshot_name = f'../snapshots/vertex_color_bg_tshirt_torch.npy'
            train_dir = '../data/tshirt_torch/train'
            test_dir = '../data/tshirt_torch/test'
            
        if self.run_type == Run_types.all_heat_paf_l2_reg_loss:
            snapshot_name = f'../snapshots/vertex_color_bg_lp_tshirt_torch.npy'
            train_dir = '../data/tshirt_torch/train'
            test_dir = '../data/tshirt_torch/test'
            
            
        if self.run_type == Run_types.all_heat_paf_kl_loss:
            snapshot_name = f'../snapshots/vertex_color_bg_kl_tshirt_torch.npy'
            train_dir = '../data/tshirt_torch/train'
            test_dir = '../data/tshirt_torch/test'
        
        return train_dir, test_dir, snapshot_name