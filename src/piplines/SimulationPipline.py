import torch
from tqdm import tqdm
import plotly.graph_objects as go
import json

from src.piplines.simulationObject import CNNSimulationStep, RNNSimulationStep, RNNIMUSimulationStep, ClassicSimulationStep, SimulationWithoutNModelStep, IMUSimulationStep, DeepVOimulationStep, UFKSimulationStep


class SimulationCNN:
    
    def __init__(self, model, device, dtrain, norm, visualization=False, loss=None, win_size=10, path_file_of_result=None):
        self.model = model
        self.device = device
        self.dtrain = dtrain
        self.norm = norm
        
        self.visualization = visualization
        self.loss = loss
        self.win_size = win_size
        self.path_file_of_result = path_file_of_result
        
    def initSimulationObject(self):
    
        if self.model is not None:
            simualtion = CNNSimulationStep(
                model=self.model,
                device=self.device,
                visualization=self.visualization,
                loss=self.loss,
                win_size=self.win_size
            )
        
        else:
            simualtion = SimulationWithoutNModelStep(
                model=self.model,
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
            
            for step, out in enumerate(train, start=1):
                simulation(out) # прочитали данные
                
                simulation.get_predict() # Предсказали
                
                predict = simulation.predict
                if self.norm:
                    predict = self.norm.denormalize(predict)
                simulation.teke_after_denorm(predict)
                
                simulation.get_trajectory_step()
                
                # if step % self.win_size == 0:
                simulation.get_metrice(step)
            
                train.set_postfix({'path_length': float(simulation.path_length),
                                    'loss': simulation.loss_pose_mean,
                                    'AT RMSE drift': simulation.p_mean,
                                    'AR RMSE drift': simulation.r_mean})
                

            self.trajectory_fact, self.trajectory = simulation.get_position()
            
            if self.path_file_of_result is not None:
                self.save_result_in_json(simulation)
    
    def save_result_in_json(self, simulation):
        
        dct_of_result = {
            'p_mean': simulation.p_mean,
            'r_mean': simulation.r_mean,
            'fact':
                {
                    'x': [x[0].item() for x in self.trajectory_fact],
                    'y': [x[1].item()  for x in self.trajectory_fact],
                    'z': [-x[2].item()  for x in self.trajectory_fact]
                }
        }
        
        if self.trajectory is not None:
            dct_of_result.update({'predict':
                    {
                        'x': [x[0].item()  for x in self.trajectory[:-1]],
                        'y': [x[1].item()  for x in self.trajectory[:-1]],
                        'z': [-x[2].item()  for x in self.trajectory[:-1]]
                    },
            })
        
        with open(self.path_file_of_result, 'w', encoding='utf-8') as f:
            json.dump(dct_of_result, f, ensure_ascii=False, indent=4)
    
    def get_pictures(self):

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=[x[0] for x in self.trajectory_fact],
            y=[x[1] for x in self.trajectory_fact],
            z=[-x[2] for x in self.trajectory_fact],
            mode='lines',
            name='Фактическая траектория',

            line=dict(
                color='royalblue',
                width=8
            )
        ))

        if self.model is not None:
            fig.add_trace(go.Scatter3d(
                x=[x[0] for x in self.trajectory[:-1]],
                y=[x[1] for x in self.trajectory[:-1]],
                z=[-x[2] for x in self.trajectory[:-1]],
                mode='lines',
                name='Предсказанная траектория',
                
                line=dict(
                    color='crimson',
                    width=8
                )
            ))

        fig.update_layout(


            width=2400,
            height=1600,

            font=dict(
                size=18,
                color='black'
            ),


            legend=dict(
                font=dict(size=18)
            ),

            scene=dict(

                xaxis=dict(
                    title='X [м]',
                    title_font=dict(size=22),
                    tickfont=dict(size=16),

                    showgrid=True,
                    gridcolor='black',
                    gridwidth=3,

                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=4,

                    showline=True,
                    linecolor='black',
                    linewidth=4,

                    backgroundcolor='white'
                ),

                yaxis=dict(
                    title='Y [м]',
                    title_font=dict(size=22),
                    tickfont=dict(size=16),

                    showgrid=True,
                    gridcolor='black',
                    gridwidth=3,

                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=4,

                    showline=True,
                    linecolor='black',
                    linewidth=4,

                    backgroundcolor='white'
                ),

                zaxis=dict(
                    title='Z [м]',
                    title_font=dict(size=22),
                    tickfont=dict(size=16),

                    showgrid=True,
                    gridcolor='black',
                    gridwidth=3,

                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=4,

                    showline=True,
                    linecolor='black',
                    linewidth=4,

                    backgroundcolor='white'
                ),

                aspectmode='data'
            )
        )

        fig.show()



class SimulationRNN(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = RNNSimulationStep(
            model=self.model,
            device=self.device,
            win_size=self.win_size
        )
        
        return simualtion       
    
class SimulationDeepVO(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = DeepVOimulationStep(
            model=self.model,
            device=self.device,
            loss=self.loss,
            win_size=self.win_size
        )
        
        return simualtion       
    
class SimulationRNNIMU(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = RNNIMUSimulationStep(
            model=self.model,
            device=self.device,
            win_size=self.win_size
        )
        
        return simualtion       
    
class SimulationIMU(SimulationCNN):
    
    def initSimulationObject(self):
        
        simualtion = IMUSimulationStep(
            model=self.model,
            device=self.device,
            win_size=self.win_size
        )
        
        return simualtion       
    
class SimulationClassic(SimulationCNN):
    
    def __init__(self, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        
    def initSimulationObject(self):
        
        simualtion = ClassicSimulationStep(
            model=self.model,
            device=self.device,
            win_size=self.win_size,
            debug=self.debug
        )
        
        return simualtion  
    
class SimulationUFK(SimulationClassic):
    
    def __init__(self, UKF, gps_window=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.UKF = UKF
        self.gps_window = gps_window

    def initSimulationObject(self):
        
        simualtion = UFKSimulationStep(
            UKF=self.UKF,
            model=self.model,
            device=self.device,
            win_size=self.win_size,
            debug=self.debug,
            gps_window=self.gps_window
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