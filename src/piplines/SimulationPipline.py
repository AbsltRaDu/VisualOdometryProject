import torch
from tqdm import tqdm
import plotly.graph_objects as go

from src.piplines.simulationObject import CNNSimulationStep, RNNSimulationStep, RNNIMUSimulationStep, ClassicSimulationStep, SimulationWithoutNModelStep, IMUSimulationStep, DeepVOimulationStep


class SimulationCNN:
    
    def __init__(self, model, device, dtrain, norm, visualization=False, loss=None):
        self.model = model
        self.device = device
        self.dtrain = dtrain
        self.norm = norm
        
        self.visualization = visualization
        self.loss = loss
        
    def initSimulationObject(self):
    
        if self.model is not None:
            simualtion = CNNSimulationStep(
                model=self.model,
                device=self.device,
                visualization=self.visualization,
                loss=self.loss
            )
        
        else:
            simualtion = SimulationWithoutNModelStep(
                device=self.device,
                visualization=self.visualization
            )
        
        return simualtion
    
    def __call__(self):
        
        simulation = self.initSimulationObject()
        train = tqdm(self.dtrain, desc=f'Работа алгоритма', position=0)
        
        if self.model is not None:
            self.model.eval()
            
        with torch.no_grad():
            
            for out in train:
                simulation(out) # прочитали данные
                
                simulation.get_predict() # Предсказали
                
                predict = simulation.predict
                if self.norm:
                    predict = self.norm.denormalize(predict)
                simulation.teke_after_denorm(predict)
                
                simulation.get_trajectory_step()
                
                if self.loss is not None:
                    train.set_postfix({'loss': simulation.loss_pose_mean})
                
            self.trajectory_fact, self.trajectory = simulation.get_position()
            
    def get_pictures(self):
        
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=[x[0] for x in self.trajectory_fact],
            y=[x[1] for x in self.trajectory_fact],
            z=[-x[2] for x in self.trajectory_fact],
            mode='lines',
            name='Фактическая траектория'
        ))

        if self.model is not None:
            fig.add_trace(go.Scatter3d(
                x=[x[0] for x in self.trajectory[:-1]],
                y=[x[1] for x in self.trajectory[:-1]],
                z=[-x[2] for x in self.trajectory[:-1]],
                mode='lines',
                name='Предсказанная траектория'
            ))

        fig.show()


class SimulationRNN(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = RNNSimulationStep(
            model=self.model,
            device=self.device
        )
        
        return simualtion       
    
class SimulationDeepVO(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = DeepVOimulationStep(
            model=self.model,
            device=self.device,
            loss=self.loss
        )
        
        return simualtion       
    
class SimulationRNNIMU(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = RNNIMUSimulationStep(
            model=self.model,
            device=self.device
        )
        
        return simualtion       
    
class SimulationIMU(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = IMUSimulationStep(
            model=self.model,
            device=self.device
        )
        
        return simualtion       
    
class SimulationClassic(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = ClassicSimulationStep(
            model=self.model,
            device=self.device
        )
        
        return simualtion  
    
    
# БЛОК 2D

from src.piplines.simulationObject import SimulationWithoutNModelStep2D, CNNSimulationStep2D, ClassicSimulationStep

class SimulationCNN2D(SimulationCNN):
    
    def initSimulationObject(self):
        
        if self.model is not None:
            simualtion = CNNSimulationStep2D(
                model=self.model,
                device=self.device,
                visualization=self.visualization
            )
            
        else:
            simualtion = SimulationWithoutNModelStep2D(
                device=self.device,
                visualization=self.visualization
            )
        
        return simualtion

    def get_pictures(self):
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[x[0] for x in self.trajectory_fact],
            y=[x[1] for x in self.trajectory_fact],
            mode='lines',
            name='Фактическая траектория'
        ))

        if self.model is not None:
            fig.add_trace(go.Scatter(
                x=[x[0] for x in self.trajectory[:-1]],
                y=[x[1] for x in self.trajectory[:-1]],
                mode='lines',
                name='Предсказанная траектория'
            ))

        fig.show()

class SimulationClassic2D(SimulationCNN2D):
    
    def initSimulationObject(self):
        
        simualtion = ClassicSimulationStep(
            model=self.model,
            device=self.device
        )
        
        return simualtion  