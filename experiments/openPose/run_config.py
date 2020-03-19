import enum
import tensorflow as tf

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
    all_heat_paf_loss = 0
    same_distance_heat_paf_loss = 1
    same_roation_heat_paf_loss = 2
    same_distance_roation_heat_paf_loss = 3
    all_paf_loss = 4
    same_distance_paf_loss = 5
    same_roation_paf_loss = 6
    same_distance_roation_paf_loss = 7
    
class Run_settings():
    def __init__(self, run_type=Run_types.all_heat_paf_loss):
        self.run_type = run_type
        
    def get_settings(self):
        snapshot_name = ''
        train_dir = ''
        test_dir = ''
        if self.run_type == Run_types.all_heat_paf_loss:
            snapshot_name = f'../snapshots/universal_noise_with_warp.npy'
            train_dir = '../data/generated/train'
            test_dir = '../data/generated/test'
            
        if self.run_type == Run_types.same_distance_heat_paf_loss:
            snapshot_name = f'../snapshots/universal_noise_with_warp_same_distance.npy'
            train_dir = '../data/sameDistance/train'
            test_dir = '../data/sameDistance/test'
            
        if self.run_type == Run_types.same_roation_heat_paf_loss:
            snapshot_name = f'../snapshots/universal_noise_with_warp_same_rotation.npy'
            train_dir = '../data/sameRotation/train'
            test_dir = '../data/sameRotation/test'

        if self.run_type == Run_types.same_distance_roation_heat_paf_loss:
            snapshot_name = f'../snapshots/universal_noise_with_warp_same_rotation_distance.npy'
            train_dir = '../data/sameRotationDistance/train'
            test_dir = '../data/sameRotationDistance/test'
            
        if self.run_type == Run_types.all_paf_loss:
            snapshot_name = f'../snapshots/universal_noise_with_warp_PAF.npy'
            train_dir = '../data/generated/train'
            test_dir = '../data/generated/test'
            
        if self.run_type == Run_types.same_distance_paf_loss:
            snapshot_name = f'../snapshots/universal_noise_with_warp_same_distance_PAF.npy'
            train_dir = '../data/sameDistance/train'
            test_dir = '../data/sameDistance/test'
            
        if self.run_type == Run_types.same_roation_paf_loss:
            snapshot_name = f'../snapshots/universal_noise_with_warp_same_rotation_PAF.npy'
            train_dir = '../data/sameRotation/train'
            test_dir = '../data/sameRotation/test'

        if self.run_type == Run_types.same_distance_roation_paf_loss:
            snapshot_name = f'../snapshots/universal_noise_with_warp_same_rotation_distance_PAF.npy'
            train_dir = '../data/sameRotationDistance/train'
            test_dir = '../data/sameRotationDistance/test'
        return train_dir, test_dir, snapshot_name
    
    def get_loss(self, estimator, target_heat, target_paf, universal_noise_tensor):
        losses_per_stage = []
                
        stage = 6
        # compute loss per stage

        paf_op = estimator.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage{stage}_L1/BiasAdd').outputs[0]
        heat_op = estimator.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage{stage}_L2/BiasAdd').outputs[0]

        # compute the loss for every channel
        loss_paf = tf.nn.l2_loss(
            tf.concat(paf_op, axis=-1) - tf.concat(target_paf, axis=-1), name='AX_loss_l1_stage%d_' % (stage)
        )
        # add heat
        if self.run_type < Run_types.all_paf_loss:
            loss_heat = tf.nn.l2_loss(
                tf.concat(heat_op, axis=-1) - tf.concat(target_heat, axis=-1), name='AX_loss_l2_stage%d_' % (stage)
            )

            # combined loss form every stage
            total_loss = tf.reduce_sum([loss_paf, loss_heat])
        else:
            total_loss = tf.reduce_sum([loss_paf])

        grad = tf.gradients(total_loss, universal_noise_tensor)[0]
        # compute the average
        grad = tf.reduce_mean(grad, 0)
        return grad, total_loss,