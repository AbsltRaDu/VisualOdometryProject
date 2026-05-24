import numpy as np
import torch
import cv2


from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT
from src.metrics.KITTI_metrics import get_KITTI_metrices

class CNNSimulationStep:
    
    def __init__(self, model, device, visualization=False, loss=None, win_size=10):

        self.model = model
        self.device = device
        
        self.trajectory = None
        self.trajectory_fact = None
        
        self.visualization = visualization
        self.loss = loss

        self.step_for_loss_trajectory = 0
        self.win_size = win_size
        self.p_mean = 0
        self.r_mean = 0
        
        self.loss_pose_mean = 0
        self.lm_count = 0
        
    def __call__(self, out):
        self.x = out[0].to(self.device)
        self.y = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        self.pose = PT.from_lie(pose)
        
        if self.visualization:
            self._visualiasation_move(self.x)
    
    def get_predict(self):
        
        self.predict = self.model(self.x)
        
        if self.loss is not None:
            self.loss(self.predict, self.y)
    
    def teke_after_denorm(self, predict):
        self.predict_denorm = predict
    
    def _visualiasation_move(self, x):
        img_left1, img_right1 = x[:, 2, :, :].squeeze().cpu().numpy(), x[:, 8, :, :].squeeze().cpu().numpy()
        img_left2, img_right2 = x[:, 5, :, :].squeeze().cpu().numpy(), x[:, 11, :, :].squeeze().cpu().numpy()

        frame = np.vstack([
            np.hstack([img_left1, img_right1]),
            np.hstack([img_left2, img_right2])
        ])

        cv2.imshow("Classic VO frames", frame)
        cv2.waitKey(1)
    
    def get_trajectory_step(self):
        
        if not self.trajectory or not self.trajectory_fact:
            pose0 = self.pose

            self.trajectory = TT.from_lie_relative(self.predict_denorm, pose0)
            self.trajectory_fact = TT.from_lie_relative(self.y, pose0)
        else:
            self.trajectory = self.trajectory.extend_lie_relative(self.predict_denorm)
            self.trajectory_fact = self.trajectory_fact.extend_lie_relative(self.y)
    
    def get_metrice(self, step):
        trajectory_fact_win = self.trajectory_fact[self.step_for_loss_trajectory:self.step_for_loss_trajectory+self.win_size]
        trajectory_pred_win = self.trajectory[self.step_for_loss_trajectory:self.step_for_loss_trajectory+self.win_size]
        
        path_lengh = trajectory_fact_win.path_length()
        motion_fact = trajectory_fact_win.relative_motion()
        motion_pred = trajectory_pred_win.relative_motion()

        r_mean_loc, p_mean_loc = get_KITTI_metrices(motion_fact, motion_pred, path_lengh) 
        
        lm = step // self.win_size
        
        self.p_mean = 1 / lm * p_mean_loc.mean().item() + (1 - 1 / lm) * self.p_mean
        self.r_mean = 1 / lm * r_mean_loc.mean().item() + (1 - 1 / lm) * self.r_mean
        
        self.step_for_loss_trajectory = step
    
    def get_position(self):

        trajectory_fact = self.trajectory_fact.positions().squeeze().cpu()
        trajectory = self.trajectory.positions().squeeze().cpu()
        
        cv2.destroyAllWindows()
        
        return trajectory_fact, trajectory
    
    
class SimulationWithoutNModelStep(CNNSimulationStep):
    
    def __init__(self, device, visualization):

        self.device = device
        self.visualization = visualization
        
        self.trajectory_fact = None
        
    def __call__(self, out):
        self.x = out[0].to(self.device)
        self.y = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        if self.visualization:
            self._visualiasation_move(self.x)
        
        self.pose = PT.from_lie(pose)
    
    def get_predict(self):
        
        self.predict = None
    
    def teke_after_denorm(self, predict):
        self.predict_denorm = predict
    
    def get_trajectory_step(self):
        
        if not self.trajectory_fact:
            pose0 = self.pose

            self.trajectory_fact = TT.from_lie_relative(self.y, pose0)
        else:
            self.trajectory_fact = self.trajectory_fact.extend_lie_relative(self.y)
    
    def get_position(self):

        trajectory_fact = self.trajectory_fact.positions().squeeze().cpu()
        
        cv2.destroyAllWindows()
        
        return trajectory_fact, None
    

class RNNSimulationStep(CNNSimulationStep):
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.hidden = None
    
    def get_predict(self):
        
        # if self.hidden is not None:
        #     self.hidden = self.hidden.to(self.device)

        predict, self.hidden = self.model(self.x.unsqueeze(dim=0), self.hidden)
        
        self.predict = predict
        
class DeepVOimulationStep(CNNSimulationStep):
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.hidden = None
    
    def get_predict(self):
        
        # if self.hidden is not None:
        #     self.hidden = self.hidden.to(self.device)

        predict, self.hidden = self.model(self.x.unsqueeze(dim=0), None)
        
        predict = PT.from_euler(predict).as_lie()
        
        self.predict = predict
        
        if self.loss is not None:
            self.lm_count += 1
            loss = self.loss(self.predict.squeeze(0), self.y)
            self.loss_pose_mean = 1 / self.lm_count * loss.item() + (1 - 1 / self.lm_count) * self.loss_pose_mean

        
class RNNIMUSimulationStep(CNNSimulationStep):
    
    def __call__(self, out):
        self.x = out[0].to(self.device)
        self.imu = out[1].to(self.device)
        self.y = out[2].to(self.device)
        pose = out[3].to(self.device)
        
        self.pose = PT.from_lie(pose)
        
    def get_predict(self):
        
        if self.hidden is not None:
            self.hidden = self.hidden.to(self.device)
        
        self.predict, self.hidden = self.model(self.x, self.imu,self.hidden)
        
class IMUSimulationStep(CNNSimulationStep):
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.velocity = torch.zeros(size=(1, 3))
        self.pred_pose = None
        self.predict = None # Просто заглушка
    
    def __call__(self, out):
        self.x = out[0].to(self.device)
        self.imu = out[1].to(self.device)
        self.imu_t = out[2].to(self.device)
        self.y = out[3].to(self.device)
        pose = out[4].to(self.device)
        
        self.pose = PT.from_lie(pose)
        
    def get_predict(self):
        
        if self.pred_pose is not None:
            self.pred_pose = self.pose
        
        self.pred_pose, self.velocity, states = self.model(self.imu, self.imu_t, self.pose, self.velocity, b_g=torch.tensor([[0.0001, 0.0001, 0.0001]]), b_a=torch.tensor([[0.0001, 0.0001, 0.0001]]))
        
    def get_trajectory_step(self):
        
        if not self.trajectory or not self.trajectory_fact:
            pose0 = self.pose

            self.trajectory = TT.from_relative(self.pred_pose, pose0)
            self.trajectory_fact = TT.from_lie_relative(self.y, pose0)
        else:
            self.trajectory = self.trajectory.extend_relative(self.pred_pose)
            self.trajectory_fact = self.trajectory_fact.extend_lie_relative(self.y)
            
        self.pred_pose = self.trajectory[-1]
        
class ClassicSimulationStep(CNNSimulationStep):
    
    def __call__(self, out):
        x = out[0].to(self.device) # Изначально прилетает 4 одноканальных изображения (1, 4, H, W)
        
        self.img_left1, self.img_right1 = x[:, 0, :, :].squeeze().numpy(), x[:, 1, :, :].squeeze().numpy()
        self.img_left2, self.img_right2 = x[:, 2, :, :].squeeze().numpy(), x[:, 3, :, :].squeeze().numpy()

        frame = np.vstack([
            np.hstack([self.img_left1, self.img_right1]),
            np.hstack([self.img_left2, self.img_right2])
        ])

        cv2.imshow("Classic VO frames", frame)
        cv2.waitKey(1)
        
        self.y = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        self.pose = PT.from_lie(pose)
    
    def get_predict(self):
        
        self.predict = self.model(self.img_left1, self.img_right1, self.img_left2).unsqueeze(0)
        
        
# БЛОК ДЛЯ 2D
from src.geometry.Pose2DTorch import Pose2DTorch as PT2D
from src.geometry.Trajectory2DTorch import Trajectory2DTorch as TT2D


class CNNSimulationStep2D(CNNSimulationStep):
    
    def __call__(self, out):
        self.x = out[0].to(self.device)
        self.y = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        self.pose = PT2D.from_vector(pose)
        
        if self.visualization:
            self._visualiasation_move(self.x)
            
    def get_trajectory_step(self):
        
        if not self.trajectory or not self.trajectory_fact:
            pose0 = self.pose

            self.trajectory = TT2D.from_vectors_relative(self.predict_denorm, pose0)
            self.trajectory_fact = TT2D.from_vectors_relative(self.y, pose0)
        else:
            self.trajectory = self.trajectory.extend_vectors_relative(self.predict_denorm)
            self.trajectory_fact = self.trajectory_fact.extend_vectors_relative(self.y)
    

class SimulationWithoutNModelStep2D(SimulationWithoutNModelStep):
    
    def __call__(self, out):
        self.x = out[0].to(self.device)
        self.y = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        if self.visualization:
            self._visualiasation_move(self.x)
        
        self.pose = PT2D.from_vector(pose)
        
    def get_trajectory_step(self):
        
        if not self.trajectory_fact:
            pose0 = self.pose

            self.trajectory_fact = TT2D.from_vectors_relative(self.y, pose0)
        else:
            self.trajectory_fact = self.trajectory_fact.extend_vectors_relative(self.y)
            
class ClassicSimulationStep2D(CNNSimulationStep2D):
    
    def __call__(self, out):
        x = out[0].to(self.device) # Изначально прилетает 4 одноканальных изображения (1, 4, H, W)
        
        self.img_left1, self.img_right1 = x[:, 0, :, :].squeeze().numpy(), x[:, 1, :, :].squeeze().numpy()
        self.img_left2, self.img_right2 = x[:, 2, :, :].squeeze().numpy(), x[:, 3, :, :].squeeze().numpy()

        frame = np.vstack([
            np.hstack([self.img_left1, self.img_right1]),
            np.hstack([self.img_left2, self.img_right2])
        ])

        cv2.imshow("Classic VO frames", frame)
        cv2.waitKey(1)
        
        self.y = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        self.pose = PT2D.from_vector(pose)
    
    def get_predict(self):
        
        self.predict = self.model(self.img_left1, self.img_right1, self.img_left2).unsqueeze(0)