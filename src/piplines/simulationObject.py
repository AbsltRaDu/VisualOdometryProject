from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT

class CNNSimulationStep:
    
    def __init__(self, model, device):

        self.model = model
        self.device = device
        
        self.trajectory = None
        self.trajectory_fact = None
        
    def __call__(self, out):
        self.x = out[0].to(self.device)
        self.y = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        self.pose = PT.from_lie(pose)
    
    def get_predict(self):
        
        self.predict = self.model(self.x)
    
    def teke_after_denorm(self, predict):
        self.predict_denorm = predict
    
    def get_trajectory_step(self):
        
        if not self.trajectory or not self.trajectory_fact:
            pose0 = self.pose
            self.trajectory = TT.from_lie_relative(self.predict_denorm, pose0)
            self.trajectory_fact = TT.from_lie_relative(self.y, pose0)
        else:
            self.trajectory = self.trajectory.extend_lie_relative(self.predict_denorm)
            self.trajectory_fact = self.trajectory_fact.extend_lie_relative(self.y)
    
    def get_position(self):
        
        trajectory_fact = self.trajectory_fact.positions().squeeze().cpu()
        trajectory = self.trajectory.positions().squeeze().cpu()
        
        
        return trajectory_fact, trajectory
    
class RNNSimulationStep(CNNSimulationStep):
    
    def __init__(self, model, device):

        super().__init__(model, device)
        self.hidden = None
    
    def get_predict(self):
        
        # if self.hidden is not None:
        #     self.hidden = self.hidden.to(self.device)

        predict, self.hidden = self.model(self.x.unsqueeze(dim=0), self.hidden)
        
        self.predict = predict

        
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