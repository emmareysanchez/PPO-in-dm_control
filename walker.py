import pickle
import scipy.signal
from dm_control import suite, viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import collections
import cv2
import yaml

# --- CARGA DE CONFIGURACIÓN ---
def load_config(path="config/ppo.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# --- ARQUITECTURA VISUAL (CNN) ---
class VisualEncoder(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Cálculo automático del tamaño de salida [cite: 20]
        with torch.no_grad():
            # Creamos un tensor de ejemplo con el tamaño de tu config
            sample_input = torch.zeros(1, input_channels, 
                                       config['environment']['observation_height'], 
                                       config['environment']['observation_width'])
            self.feature_dim = self.conv(sample_input).shape[1]

    def forward(self, x):
        return self.conv(x)


class policy_net(nn.Module):
    def __init__(self, input_channels, n_actions, hdim=256):
        super().__init__()
        self.encoder = VisualEncoder(input_channels)
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, n_actions) # Un logit por cada acción del dqn.yaml
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x) # Salida: Probabilidades sin activar (logits)

class value_net(nn.Module):
    def __init__(self, input_channels, hdim=256):
        super().__init__()
        self.encoder = VisualEncoder(input_channels)
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)



# --- ALGORITMO PPO ---
class PPO:
    def __init__(self, input_channels, n_actions):
        params = config['ppo']
        self.device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.action_prototypes = config['environment']['action_prototypes']

        self.actor = policy_net(input_channels, n_actions, config['architecture']['hidden_dim']).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params['actor_lr'])

        self.critic = value_net(input_channels, config['architecture']['hidden_dim']).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params['critic_lr'])

        self.gamma = params['gamma']
        self.lambd = params['lambd']
        self.K_epochs = params['k_epochs']
        self.eps_clip = params['eps_clip']
        self.entropy_coef = params.get('entropy_coef', 0.01)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)
        
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        
        # Mapeo al vector físico: [hip, knee, ankle...]
        u_vector = np.array(self.action_prototypes[action_idx], dtype=np.float32)
        
        return action_idx, u_vector

    def update(self, buffer):
        s = torch.tensor(np.array(buffer['x']), dtype=torch.float).to(self.device)
        a = torch.tensor(np.array(buffer['u']), dtype=torch.long).view(-1, 1).to(self.device)
        r = torch.tensor(np.array(buffer['r']), dtype=torch.float).view(-1, 1).to(self.device)
        next_s = torch.tensor(np.array(buffer['next_x']), dtype=torch.float).to(self.device)
        done = torch.tensor(np.array(buffer['done']), dtype=torch.float).view(-1, 1).to(self.device)

        # Cálculo de Ventaja (GAE)
        with torch.no_grad():
            td_target = r + self.gamma * self.critic(next_s) * (1 - done)
            delta = td_target - self.critic(s)
        
        advantage = []
        adv = 0.0
        for d in reversed(delta.cpu().numpy()):
            adv = d + self.gamma * self.lambd * adv
            advantage.append(adv)
        advantage.reverse()
        advantage = torch.tensor(advantage, dtype=torch.float).to(self.device)

        # Log probs antiguos para el ratio de PPO [cite: 39, 41]
        with torch.no_grad():
            old_logits = self.actor(s)
            old_probs = F.softmax(old_logits, dim=-1)
            old_log_probs = torch.distributions.Categorical(old_probs).log_prob(a.squeeze()).view(-1, 1)

        for _ in range(self.K_epochs):
            logits = self.actor(s)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            log_probs = dist.log_prob(a.squeeze()).view(-1, 1)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = F.mse_loss(self.critic(s), td_target)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# --- UTILIDADES DE IMAGEN ---
def get_screen(env):
    # Captura RGB y convierte a gris para eficiencia [cite: 37, 79]
    screen = env.physics.render(height=config['environment']['observation_height'], 
                                width=config['environment']['observation_width'], 
                                camera_id=0)
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    return screen.astype(np.float32) / 255.0

def rollout(env, agent, T=1000):
    episode_return = 0
    # 'u' almacenará el índice de la acción para la Categorical de PPO
    episode_traj = {'x': [], 'u': [], 'next_x': [], 'r': [], 'done': []}
    
    # Reiniciar el entorno de dm_control
    timestep = env.reset() [cite: 22]
    
    # Configuración de frames y repetición de acción desde el YAML
    stack_size = config['environment']['frame_stack']
    action_repeat = config['environment'].get('action_repeat', 1)
    
    # Inicializar el stack de frames con la imagen inicial
    current_frame = get_screen(env)
    frame_stack = collections.deque([current_frame] * stack_size, maxlen=stack_size)
    
    # Estado inicial: [Channels, H, W]
    x = np.stack(frame_stack, axis=0)

    for _ in range(T):
        # El agente selecciona el índice (discreto) y el vector físico asociado
        action_idx, u_vector = agent.select_action(x)
        
        # Aplicar la acción con repetición (action skipping) para estabilidad
        cumulative_reward = 0
        for _ in range(action_repeat):
            timestep = env.step(u_vector) 
            cumulative_reward += timestep.reward
            done = timestep.last()
            if done:
                break
        
        # Capturar el nuevo frame y actualizar el stack
        new_frame = get_screen(env) 
        frame_stack.append(new_frame)
        next_x = np.stack(frame_stack, axis=0)
        
        # Guardar la transición en la trayectoria
        episode_traj['x'].append(x)
        episode_traj['u'].append(action_idx) # Guardamos el índice para PPO update
        episode_traj['next_x'].append(next_x)
        episode_traj['r'].append(cumulative_reward)
        episode_traj['done'].append(done)

        episode_return += cumulative_reward
        x = next_x
        
        if done:
            break

    return episode_traj, episode_return

# --- ENTRENAMIENTO ---
def train(env, agent, num_episodes):
    return_list = []
    for i in range(num_episodes):
        episode_traj, episode_return = rollout(env, agent)
        return_list.append(episode_return)
        agent.update(episode_traj)

        if (i+1) % 10 == 0:
            print(f'Episodio {i+1}: Retorno {episode_return:.2f}')
        
        if (i + 1) % config['logging']['save_freq'] == 0:
            torch.save(agent.actor.state_dict(), f'ppo_actor_{i + 1}.pth')
            print(f'Checkpoint guardado en episodio {i+1}')

    return agent

if __name__ == '__main__':
    # Cargar el entorno de MuJoCo 
    env = suite.load(config['environment']['domain'], config['environment']['task'])
    
    # IMPORTANTE: n_actions es el tamaño de tu lista de prototipos discretos
    n_actions = len(config['environment']['action_prototypes'])
    input_channels = config['environment']['frame_stack']

    agent = PPO(input_channels, n_actions)
    
    # Iniciar entrenamiento (4 sesiones recomendadas) 
    agent = train(env, agent, 5000)