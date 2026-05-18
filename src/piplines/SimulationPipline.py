import torch
from tqdm import tqdm
import plotly.graph_objects as go

from src.piplines.simulationObject import CNNSimulationStep, RNNSimulationStep, RNNIMUSimulationStep



class SimulationCNN:
    
    
    def __init__(self, model, device, dtrain, norm):
        self.model = model
        self.device = device
        self.dtrain = dtrain
        self.norm = norm
        
    def initSimulationObject(self):
        
        simualtion = CNNSimulationStep(
            model=self.model,
            device=self.device
        )
        
        return simualtion
    
    def __call__(self):
        
        simulation = self.initSimulationObject()
        train = tqdm(self.dtrain, desc=f'Работа алгоритма', position=0)
        
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
                
            self.trajectory_fact, self.trajectory = simulation.get_position()
            
    def get_pictures(self):
        
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=[x[0] for x in self.trajectory_fact],
            y=[x[1] for x in self.trajectory_fact],
            z=[x[2] for x in self.trajectory_fact],
            mode='lines',
            name='Фактическая траектория'
        ))

        fig.add_trace(go.Scatter3d(
            x=[x[0] for x in self.trajectory[:-1]],
            y=[x[1] for x in self.trajectory[:-1]],
            z=[x[2] for x in self.trajectory[:-1]],
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
    
class SimulationRNNIMU(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = RNNIMUSimulationStep(
            model=self.model,
            device=self.device
        )
        
        return simualtion       
    