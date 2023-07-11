from env.FlightModel import FlightModel
from env.FlightEnv import PlaneEnvironment
from utils import runner
from tensorforce import Agent
import matplotlib.pyplot as plt


# Instantiate our Flight Model
# FlightModel = FlightModel()
# Instantiane our environment
mediaRepeticoes = 1 # número de vezes que o ambiente será reiniciado
repeticoes = 1 # número de episódios
n_episodes = 50 #número de passos por episódio
max_step_per_episode = 1000000
nnearest = 2

altura = 1000
largura = 1000
fig, ax = plt.subplots()

quantAeronaves = 5
quantVertiports = 30
toleranciaLinear = 100  # metros distância que será usada para determinar se há conflito
# ax.axis([-1, altura, -1, largura])
# ax.axis([0, mediaRepeticoes+1, 0, 200])

# change the fontsize
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)



environment = PlaneEnvironment(n_episodes, max_step_per_episode, ax, 0, repeticoes, altura, largura, quantAeronaves,
                 quantVertiports, toleranciaLinear, nnearest, False)
# Instantiate a Tensorforce agent


agent = Agent.load('model/2023-04-10 19:09:17.325329')

# #candidatos a agente: constant, reinforce
# agent = Agent.create(
#         agent='dqn', environment=environment,
#         # batch_size=10,
#         # memory=15000000
#         # Automatically configured network

# agent = Agent.create(
#         agent='ddqn', environment=environment,
#         # Automatically configured network
#         # network='auto',
#         # Optimization
#         batch_size=10,
#         # update=10,
#         # optimizer='adam',
#         # objective='policy_gradient',
#         # reward_estimation=dict(horizon='episode'),
#         # use_beta_distribution=True, #melhora a distância final sem afetar os outros resultados
#         memory=15000000,
#         # learning_rate=0.0001,
#         # discount=0.99,
#         # multi_step=100
#
#         # update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
#         # multi_step=5,
#         # # Reward estimation
#         # likelihood_ratio_clipping=0.2, , predict_terminal_values=False,
#         # # Critic
#         # baseline='auto',
#         # baseline_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
#         # # Preprocessing
#         # preprocessing=None,
#         # # Exploration
#         # exploration=0.0, variable_noise=0.0,
#         # # Regularization
#         # l2_regularization=0.0, entropy_regularization=0.0,
#         # # TensorFlow etc
#         # name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
#         # summarizer=None, recorder=None
#
#
#     )
# agent = Agent.create(agent="ppo",environment=environment)



# Call runner
runner(
    agent,
    max_step_per_episode,
    n_episodes, ax, fig, repeticoes, mediaRepeticoes, altura, largura, quantAeronaves, quantVertiports, toleranciaLinear, nnearest)