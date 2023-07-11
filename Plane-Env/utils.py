import os
import time
import itertools
import shutil
from collections import namedtuple
import numpy as np
import math
from tensorforce import Agent
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from env.FlightEnv import PlaneEnvironment
import matplotlib.animation as animation
import datetime


# def write_to_txt_general(data, path):
#     cur_path = os.path.dirname(__file__)
#     new_path = os.path.relpath("env/" + str(path), cur_path)
#     text_file = open(new_path, "w")
#     n = text_file.write(str(data))
#     text_file.close()
#
#
# def write_pos_and_angles_to_txt(environment, path):
#     write_to_txt_general(environment.FlightModel.Pos_vec, path + "/positions.txt")
#     write_to_txt_general(environment.FlightModel.theta_vec, path + "/angles.txt")
#
#
# def write_combination_to_txt(param_dict, folder=None):
#     cur_path = os.path.dirname(__file__)
#     if folder:
#         new_path = os.path.join("env", "Graphs", str(folder), "params.txt")
#     else:
#         new_path = os.path.relpath("env/params.txt", cur_path)
#
#     text_file = open(new_path, "w")
#     n = text_file.write(str(param_dict))
#     text_file.close()


def create_agent(param_grid, i, directory, environment):
    return Agent.create(
        agent="ppo",
        environment=environment,
        # Automatically configured network
        network=dict(
            type=param_grid["network"],
            size=param_grid["size"],
            depth=param_grid["depth"],
        ),
        # Optimization
        batch_size=param_grid["batch_size"],
        update_frequency=param_grid["update_frequency"],
        learning_rate=param_grid["learning_rate"],
        subsampling_fraction=param_grid["subsampling_fraction"],
        optimization_steps=param_grid["optimization_steps"],
        # Reward estimation
        likelihood_ratio_clipping=param_grid["likelihood_ratio_clipping"],
        discount=param_grid["discount"],
        estimate_terminal=param_grid["estimate_terminal"],
        # Critic
        critic_network="auto",
        critic_optimizer=dict(
            optimizer="adam",
            multi_step=param_grid["multi_step"],
            learning_rate=param_grid["learning_rate_critic"],
        ),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=param_grid["exploration"],
        variable_noise=param_grid["variable_noise"],
        # Regularization
        l2_regularization=param_grid["l2_regularization"],
        entropy_regularization=param_grid["entropy_regularization"],
        # TensorFlow etc
        name="agent_" + str(i),
        device=None,
        parallel_interactions=5,
        seed=124,
        execution=None,
        recorder=dict(directory=directory, frequency=1000),
        summarizer=None,
        saver=dict(directory=directory, filename="agent_" + str(i)),
    )


# def gridsearch_tensorforce(
#     environment, param_grid_list, max_step_per_episode, n_episodes
# ):
#     GridSearch = namedtuple("GridSearch", ["scores", "names"])
#     gridsearch = GridSearch([], [])
#
#     # Compute the different parameters combinations
#     param_combinations = list(itertools.product(*param_grid_list.values()))
#     for i, params in enumerate(param_combinations, 1):
#         if not os.path.exists(os.path.join("env", "Graphs", str(i))):
#             os.mkdir(os.path.join("env", "Graphs", str(i)))
#         # fill param dict with params
#         param_grid = {
#             param_name: params[param_index]
#             for param_index, param_name in enumerate(param_grid_list)
#         }
#         directory = os.path.join(os.getcwd(), "env", "Models", str(i))
#         if os.path.exists(directory):
#             shutil.rmtree(directory, ignore_errors=True)
#
#         agent = create_agent(param_grid, i, directory, environment)
#         # agent = Agent.load(directory="data/checkpoints")
#         gridsearch.scores.append(
#             trainer(
#                 environment,
#                 agent,
#                 max_step_per_episode,
#                 n_episodes,
#                 combination=i,
#                 total_combination=len(param_combinations),
#             )
#         )
#         store_results_and_graphs(i, environment, param_grid)
#         gridsearch.names.append(str(param_grid))
#     dict_scores = dict(zip(gridsearch.names, gridsearch.scores))
#     write_to_txt_general(dict_scores, "results.txt")
#     best_model = min(dict_scores, key=dict_scores.get)
#     print(
#         "best model",
#         best_model,
#         "number",
#         np.argmin(gridsearch.scores),
#         "score",
#         dict_scores[best_model],
#     )
#
#
# def store_results_and_graphs(i, environment, param_grid):
#     write_pos_and_angles_to_txt(environment, "")
#     write_combination_to_txt(param_grid, folder=str(i))
#
#
# def show_policy(thrust_vec, theta_vec, distances, combination, title="Policy vs time"):
#     plot_duo(
#         Series=[thrust_vec, theta_vec],
#         labels=["Thrust", "Theta"],
#         xlabel="time (s)",
#         ylabel="Force intensity (N)/Angle value (°)",
#         title=title,
#         save_fig=True,
#         path="env",
#         folder=str(combination),
#         time=True,
#     )
#
#     plot_multiple(
#         Series=[distances],
#         labels=["TO-Distance"],
#         xlabel="episodes",
#         ylabel="TO-Distance (m)",
#         title="Distance vs episodes",
#         save_fig=True,
#         path="env",
#         folder=str(combination),
#         time=False,
#     )
#
#
# def train_info(i, n_episodes, start_time, combination):
#     temp_time = time.time() - start_time
#     time_per_episode = temp_time / (i + 1)
#     print(
#         "combination : ",
#         combination,
#         "episode : ",
#         i,
#         "/",
#         n_episodes,
#         " time per episode",
#         round(time_per_episode, 2),
#         "seconds. ",
#         "estimated time to finish",
#         int((time_per_episode * n_episodes) - temp_time),
#         "seconds.",
#     )
#
#
# def terminal_info(episode, states, actions):
#     print("actions", actions, "states", states)
#     print(
#         "mean reward",
#         np.mean(episode.rewards),
#         "mean action",
#         round(np.mean(episode.thrust_values), 2),
#         round(np.mean(episode.theta_values), 2),
#         "std",
#         round(np.std(episode.thrust_values), 2),
#         round(np.std(episode.theta_values), 2),
#         "episode length",
#         len(episode.rewards),
#     )


# def run(
#     environment,
#     agent,
#     n_episodes,
#     max_step_per_episode,
#     combination,
#     total_combination,
#     batch,
#     test=False,
# ):
#     """
#     Train agent for n_episodes
#     """
#     environment.FlightModel.max_step_per_episode = max_step_per_episode
#     Score = namedtuple("Score", ["reward", "reward_mean", "distance"])
#     score = Score([], [], [])
#
#     start_time = time.time()
#     for i in range(1, n_episodes + 1):
#         print("Episódio %d" %(i))
#         Episode = namedtuple("Episode", ["rewards", "thrust_values", "theta_values"],)
#         episode = Episode([], [], [])
#
#         if total_combination == 1 and (
#             i % 50 == 0
#         ):  # Print training information every 50 episodes
#             train_info(i, n_episodes, start_time, combination)
#
#         # Initialize episode
#         states = environment.reset()
#         internals = agent.initial_internals()
#         terminal = False
#
#         while not terminal:  # While an episode has not yet terminated
#
#             if test:  # Test mode (deterministic, no exploration)
#                 actions, internals = agent.act(
#                     states=states, internals=internals, evaluation=True
#                 )
#                 states, terminal, reward = environment.execute(actions=actions)
#             else:  # Train mode (exploration and randomness)
#                 actions = agent.act(states=states)
#                 states, terminal, reward = environment.execute(actions=actions)
#                 agent.observe(terminal=terminal, reward=reward)
#
#             episode.thrust_values.append(round(actions["thrust"], 2))
#             episode.theta_values.append(round(actions["theta"], 2))
#             episode.rewards.append(reward)
#             # if terminal and (i % 100 == 0):
#             #     terminal_info(
#             #         episode, states, actions,
#             #     )
#         score.reward.append(np.sum(episode.rewards))
#         score.reward_mean.append(np.mean(score.reward))
#         score.distance.append(environment.FlightModel.Pos[0])
#     if not (test):
#         show_policy(
#             episode.thrust_values,
#             episode.theta_values,
#             score.distance,
#             combination,
#             title="pvt_train_" + str(batch),
#         )
#     if test:
#         show_policy(
#             episode.thrust_values,
#             episode.theta_values,
#             score.distance,
#             combination,
#             title="pvt_" + str(batch),
#         )
#         if not os.path.exists(os.path.join("env", "Pos_and_angles", str(batch))):
#             os.mkdir(os.path.join("env", "Pos_and_angles", str(batch)))
#         write_pos_and_angles_to_txt(environment, "Pos_and_angles/" + str(batch))
#     # plot_multiple(
#     #     Series=[score.reward, score.reward_mean],
#     #     labels=["Reward", "Mean reward"],
#     #     xlabel="time (s)",
#     #     ylabel="Reward",
#     #     title="Global Reward vs time",
#     #     save_fig=True,
#     #     path="env",
#     #     folder=str(combination),
#     # )
#     return environment.FlightModel.Pos[0]


# def batch_information(
#     i, result_vec, combination, total_combination, temp_time, number_batches
# ):
#     if result_vec:
#
#         print(
#             "Combination {}/{}, Batch {}/{}, Best result: {},Time per batch {}s, Combination ETA: {}mn{}s, Total ETA: {}mn{}s".format(
#                 combination,
#                 total_combination,
#                 i,
#                 number_batches,
#                 int(result_vec[-1]),
#                 round(temp_time / i, 1),
#                 round(((temp_time * number_batches / i) - temp_time) // 60),
#                 round(((temp_time * number_batches / i) - temp_time) % 60),
#                 round(((temp_time * number_batches / i) * total_combination) // 60),
#                 round(((temp_time * number_batches / i) * total_combination) % 60),
#             )
#         )
#
#
# def trainer(
#     environment,
#     agent,
#     max_step_per_episode,
#     n_episodes,
#     n_episodes_test=1,
#     combination=1,
#     total_combination=1,
# ):
#
#     result_vec = []
#     start_time = time.time()
#     number_batches = round(n_episodes / 100) + 1
#     for i in range(1, number_batches):
#         temp_time = time.time() - start_time
#         batch_information(
#             i, result_vec, combination, total_combination, temp_time, number_batches
#         )
#         # Train agent
#         run(
#             environment,
#             agent,
#             100,
#             max_step_per_episode,
#             combination=combination,
#             total_combination=total_combination,
#             batch=i,
#         )
#         # Test Agent
#         result_vec.append(
#             run(
#                 environment,
#                 agent,
#                 n_episodes_test,
#                 max_step_per_episode,
#                 combination=combination,
#                 total_combination=total_combination,
#                 batch=i,
#                 test=True,
#             )
#         )
#     environment.FlightModel.plot_graphs(save_figs=True, path="env")
#     plot_multiple(
#         Series=[result_vec],
#         labels=["TO-Distance"],
#         xlabel="episodes",
#         ylabel="Distance (m)",
#         title="TO-Distance vs episodes",
#         save_fig=True,
#         path="env",
#         folder=str(combination),
#         time=False,
#     )
#     agent.close()
#     environment.close()
#     save_distances(
#         result_vec, combination, environment
#     )  # saves distances results for each combination in a txt file.
#     return environment.FlightModel.Pos[0]
#
#
# def save_distances(result_vec, combination, environment):
#     """
#     Saves distances results in a txt in the current combination folder
#     """
#     if not os.path.exists(os.path.join("env", "Distances", str(combination))):
#         os.mkdir(os.path.join("env", "Distances", str(combination)))
#     write_to_txt_general(result_vec, "Distances/" + str(combination) + "/distances.txt")
#     write_pos_and_angles_to_txt(environment, "Distances/" + str(combination))



def runner(
    agent,
    max_step_per_episode,
    n_episodes,
    ax,
    fig,
    repeticoes,
    mediaRepeticoes,
    altura, largura, quantAeronaves,
    quantVertiports, toleranciaLinear, nnearest
):
    # Train agent
    result_vec = [] #initialize the result list
    # for i in range(n_episodes): #Divide the number of episodes into batches of 100 episodes
    #     print("loop do runner. i=%d" %(i))
    #     if result_vec:
    #         print("batch", i, "Best result", result_vec[-1]) #Show the results for the current batch
    #     # Train Agent for 100 episode
    run(agent, n_episodes, max_step_per_episode, ax, fig, repeticoes, mediaRepeticoes, altura, largura, quantAeronaves,
                 quantVertiports, toleranciaLinear, nnearest)
        # # Test Agent for this batch
        # test_results = run(
        #         environment,
        #         agent,
        #         n_episodes_test,
        #         max_step_per_episode,
        #         #combination=combination,
        #         test=True
        #     )
        #
        # # Append the results for this batch
        # result_vec.append(test_results)
    # Plot the evolution of the agent over the batches
    # plot_multiple(
    #     Series=[result_vec],
    #     labels = ["Reward"],
    #     xlabel = "episodes",
    #     ylabel = "Reward",
    #     title = "Reward vs episodes",
    #     save_fig=True,
    #     path="env",
    #     folder=str(combination),
    #     time=False,
    # )
    #Terminate the agent and the environment
    agent.close()

def preencheMatrizMedias(matrizMedias, coluna, arrayMediaDist, mediaRepeticoes):


    #calcular as médias para inserir no gráfico
    q0soma, q1soma, q2soma, q3soma, q4soma = 0, 0, 0, 0, 0
    for mov in range(mediaRepeticoes):
        # print("mov=%d" %(mov))
        # lista = mediaDistConflitos[mov]
        q0soma += arrayMediaDist[mov][0]
        q1soma += arrayMediaDist[mov][1]
        q2soma += arrayMediaDist[mov][2]
        q3soma += arrayMediaDist[mov][3]
        q4soma += arrayMediaDist[mov][4]
    # print("q2soma após o for: %f" %(q2soma))

    matrizMedias[0][coluna] = q0soma/mediaRepeticoes
    matrizMedias[1][coluna] = q1soma/mediaRepeticoes
    matrizMedias[2][coluna] = q2soma/mediaRepeticoes
    matrizMedias[3][coluna] = q3soma/mediaRepeticoes
    matrizMedias[4][coluna] = q4soma/mediaRepeticoes

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def inverteMatriz(distSimulacoes, repeticoes ):
    matrizRetorno = np.empty([5, repeticoes], dtype='f')
    for i in range(repeticoes):
        print("invertendo matriz. i= %d "%(i))
        if distSimulacoes[i]:
            matrizRetorno[0][i] = distSimulacoes[i][0]
            matrizRetorno[1][i] = distSimulacoes[i][1]
            matrizRetorno[2][i] = distSimulacoes[i][2]
            matrizRetorno[3][i] = distSimulacoes[i][3]
            matrizRetorno[4][i] = distSimulacoes[i][4]
        else:
            matrizRetorno[0][i] = 0
            matrizRetorno[1][i] = 0
            matrizRetorno[2][i] = 0
            matrizRetorno[3][i] = 0
            matrizRetorno[4][i] = 0
    return matrizRetorno


def run(agent, nb_timesteps, max_step_per_episode, ax, fig, repeticoes, mediaRepeticoes, altura, largura, quantAeronaves,
                 quantVertiports, toleranciaLinear, nnearest, test=False):
    """
    Train agent for n_episodes
    """
    # environment.FlightModel.max_step_per_episode = max_step_per_episode
    # Loop over episodes
    matrizMedias = np.empty([5, 4], dtype='f')
    matrizDistFinais = np.empty([5, 4], dtype='f')

    mediaDistSimulacoes = np.empty(mediaRepeticoes, dtype=object)
    distSimulacoes = np.empty(repeticoes, dtype=object)
    distPassosSimulacao = []
    distFinalDRL = []

    mediaDistSimulacoesAvoid = np.empty(mediaRepeticoes, dtype=object)
    distSimulacoesAvoid = np.empty(repeticoes, dtype=object)
    distPassosSimulacaoAvoid = []
    distFinalAvoid = []

    mediaDistSimulacoesFree = np.empty(mediaRepeticoes, dtype=object)
    distSimulacoesFree = np.empty(repeticoes, dtype=object)
    distPassosSimulacaoFree = []
    distFinalFree = []

    mediaDistSimulacoesRandom= np.empty(mediaRepeticoes, dtype=object)
    distSimulacoesRandom = np.empty(repeticoes, dtype=object)
    distPassosSimulacaoRandom = []
    distFinalRandom = []

    quantMomentosMonitor = 0
    quantMomentosConflito = 0
    quantMomentosCritico = 0
    quantMomentosMonitorAvoid = 0
    quantMomentosConflitoAvoid = 0
    quantMomentosCriticoAvoid = 0
    quantMomentosMonitorFree = 0
    quantMomentosConflitoFree = 0
    quantMomentosCriticoFree = 0
    quantMomentosMonitorRandom = 0
    quantMomentosConflitoRandom = 0
    quantMomentosCriticoRandom = 0
    diagonal = math.sqrt(toleranciaLinear**2 + toleranciaLinear**2)



    terminal = False
    debug = False





    momentosCriticoRepeticao = [] #array de tamanho mediaRepeticoes
    momentosConflitoRepeticao = [] #array de tamanho mediaRepeticoes

    momentosCriticoRepeticaoAvoid = []
    momentosConflitoRepeticaoAvoid = []

    momentosCriticoRepeticaoFree = []
    momentosConflitoRepeticaoFree = []

    momentosCriticoRepeticaoRandom = []
    momentosConflitoRepeticaoRandom = []
    # internals = agent.initial_internals()
    print("terminou o reset do environment")
    for k in range (mediaRepeticoes):
        momentosCriticoEpisodio = []  # array de tamanho repeticoes que vai guardar todos os rewardsEpisodio
        momentosConflitoEpisodio = []  # array de tamanho repeticoes que vai guardar todos os menorDistanciaEpisodio

        momentosCriticoEpisodioAvoid = []
        momentosConflitoEpisodioAvoid = []

        momentosCriticoEpisodioFree = []
        momentosConflitoEpisodioFree = []

        momentosCriticoEpisodioRandom = []
        momentosConflitoEpisodioRandom = []

        for j in range(repeticoes):
            print("*************************************************************************************")
            print("k = %d, repeticão = %d" % (k, j))
            print("*************************************************************************************")
            agent.environment = PlaneEnvironment(nb_timesteps, max_step_per_episode, ax, repeticoes, j, altura, largura, quantAeronaves,
                 quantVertiports, toleranciaLinear, nnearest, True, debug)
            # agent.environment.reset(nb_timesteps, ax, repeticoes, j, debug)
            states = np.zeros(shape=(agent.environment.STATES_SIZE,))
            grafMomentosCriticos = []  # array de tamanho nb_timesteps que vai guardar todos os rewards de um episódio
            grafMomentosConflito = []  # array de tamanho nb_timesteps que vai guardar a menor distância a cada passo do episódio
            i = 0
            for i in range(nb_timesteps):
                if debug:
                    print("****")
                    print(" loop do run. i = %d" % (i))
                    print("****")
                # Initialize episode
                episode_length = 0

                # while not terminal:
                # Run episode
                episode_length += 1
                if debug:
                    print("dentro do loop interno do run. episode_length=%d" % (episode_length))
                    print("states sobre o qual vai agir:")
                    print(states)
                actions = agent.act(states=states)
                states, terminal, reward, distAvoid, distFree, distRandom, arrayDist = agent.environment.execute(i, debug, ax, nb_timesteps, actions=actions )
                # if debug:
                #    print("reward que será passado: %f" %(reward))
                agent.observe(terminal=terminal, reward=reward)
                # print("vai observar os states para saber a distância do intruso:")
                # print(states)

                grafMomentosCriticos.append(reward)
                distanciaIntruso = states[0]
                grafMomentosConflito.append(distanciaIntruso)



                # preencher os arrays que registram a separação mínima
                if(i > nb_timesteps/10): # desconsiderar o primeiro décimo da simulação para evitar conflitos onde as duas aeronaves decolam de vertiports muito próximos.


                    # DRL
                    if (distanciaIntruso < diagonal):  # a distância abaixo desse valor indica que há um conflito real
                        distPassosSimulacao.append(distanciaIntruso)
                        if debug:
                            print("vai incrementar o contador DRL. valor atual= %d" % (quantMomentosMonitor))
                        quantMomentosMonitor += 1
                        if distanciaIntruso < toleranciaLinear / 2:
                            if debug:
                                print("vai incrementar o contador de conflitos DRL. valor atual= %d" % (
                                    quantMomentosConflito))
                            quantMomentosConflito += 1
                            if distanciaIntruso < toleranciaLinear / 4:
                                quantMomentosCritico +=1

                    #AVOID
                    #verificar se o contador de separações será preenchido
                    if distAvoid < diagonal:
                        distPassosSimulacaoAvoid.append(distAvoid)
                        if debug:
                            print("vai incrementar o contador Avoid. valor atual= %d" %(quantMomentosMonitorAvoid))
                        quantMomentosMonitorAvoid += 1
                        if distAvoid < toleranciaLinear / 2:
                            if debug:
                                print("vai incrementar o contador de conflitos Avoid. valor atual= %d" % (quantMomentosConflitoAvoid))
                            quantMomentosConflitoAvoid += 1
                            if distAvoid < toleranciaLinear / 4:
                                quantMomentosCriticoAvoid +=1

                    # FREE
                    # verificar se o contador de separações será preenchido
                    if distFree < diagonal:
                        distPassosSimulacaoFree.append(distFree)
                        if debug:
                            print("vai incrementar o contador Free. valor atual= %d" % (quantMomentosMonitorFree))
                        quantMomentosMonitorFree += 1
                        if distFree < toleranciaLinear / 2:
                            if debug:
                                print("vai incrementar o contador de conflitos Free. valor atual= %d" % (
                                    quantMomentosConflitoFree))
                            quantMomentosConflitoFree += 1
                            if distFree < toleranciaLinear / 4:
                                quantMomentosCriticoFree +=1

                    # RANDOM
                    # verificar se o contador de separações será preenchido
                    if distRandom < diagonal:
                        distPassosSimulacaoRandom.append(distRandom)
                        if debug:
                            print("vai incrementar o contador Random. valor atual= %d. distância calculada: %f" % (quantMomentosMonitorRandom, distRandom))
                        quantMomentosMonitorRandom += 1
                        if distRandom < toleranciaLinear / 2:
                            if debug:
                                print("vai incrementar o contador de conflitos Random. valor atual= %d" % (
                                    quantMomentosConflitoRandom))
                            quantMomentosConflitoRandom += 1
                            if distRandom < toleranciaLinear / 4:
                                quantMomentosCriticoRandom +=1

            momentosConflitoEpisodio.append(quantMomentosConflito)
            quantMomentosConflito = 0 #zerar a variável para o próximo episódio
            momentosCriticoEpisodio.append(quantMomentosCritico)
            quantMomentosCritico = 0  # zerar a variável para o próximo episódio

            momentosConflitoEpisodioAvoid.append(quantMomentosConflitoAvoid)
            quantMomentosConflitoAvoid = 0  # zerar a variável para o próximo episódio
            momentosCriticoEpisodioAvoid.append(quantMomentosCriticoAvoid)
            quantMomentosCriticoAvoid = 0  # zerar a variável para o próximo episódio

            momentosConflitoEpisodioFree.append(quantMomentosConflitoFree)
            quantMomentosConflitoFree = 0  # zerar a variável para o próximo episódio
            momentosCriticoEpisodioFree.append(quantMomentosCriticoFree)
            quantMomentosCriticoFree = 0  # zerar a variável para o próximo episódio

            momentosConflitoEpisodioRandom.append(quantMomentosConflitoRandom)
            quantMomentosConflitoRandom = 0  # zerar a variável para o próximo episódio
            momentosCriticoEpisodioRandom.append(quantMomentosCriticoRandom)
            quantMomentosCriticoRandom = 0  # zerar a variável para o próximo episódio

            # colocar a média desta repetição no array geral

            if distPassosSimulacao: #só vai avançar se o array não for vazio
                max_value = max(distPassosSimulacao)
                min_value = min(distPassosSimulacao)
                quarters = np.percentile(distPassosSimulacao, [25, 50, 75])
                distSimulacoes[j] = [min_value , quarters [0], quarters[1], quarters [2], max_value]
                matrizMedias[0][0] = min_value
                matrizMedias[1][0] = quarters [0]
                matrizMedias[2][0] = quarters [1]
                matrizMedias[3][0] = quarters [2]
                matrizMedias[4][0] = max_value

            if distPassosSimulacaoAvoid:
                max_valueAvoid = max(distPassosSimulacaoAvoid)
                min_valueAvoid = min(distPassosSimulacaoAvoid)
                quartersAvoid = np.percentile(distPassosSimulacaoAvoid, [25, 50, 75])
                distSimulacoesAvoid[j] = [min_valueAvoid, quartersAvoid[0], quartersAvoid[1], quartersAvoid[2], max_valueAvoid]
                matrizMedias[0][1] = min_valueAvoid
                matrizMedias[1][1] = quartersAvoid[0]
                matrizMedias[2][1] = quartersAvoid[1]
                matrizMedias[3][1] = quartersAvoid[2]
                matrizMedias[4][1] = max_valueAvoid

            if distPassosSimulacaoFree:
                max_valueFree = max(distPassosSimulacaoFree)
                min_valueFree = min(distPassosSimulacaoFree)
                quartersFree = np.percentile(distPassosSimulacaoFree, [25, 50, 75])
                distSimulacoesFree[j] = [min_valueFree, quartersFree[0], quartersFree[1], quartersFree[2],
                                          max_valueFree]
                matrizMedias[0][2] = min_valueFree
                matrizMedias[1][2] = quartersFree[0]
                matrizMedias[2][2] = quartersFree[1]
                matrizMedias[3][2] = quartersFree[2]
                matrizMedias[4][2] = max_valueFree

            if distPassosSimulacaoRandom:
                max_valueRandom = max(distPassosSimulacaoRandom)
                min_valueRandom = min(distPassosSimulacaoRandom)
                quartersRandom = np.percentile(distPassosSimulacaoRandom, [25, 50, 75])
                distSimulacoesRandom[j] = [min_valueRandom, quartersRandom[0], quartersRandom[1], quartersRandom[2],
                                         max_valueRandom]
                matrizMedias[0][3] = min_valueRandom
                matrizMedias[1][3] = quartersRandom[0]
                matrizMedias[2][3] = quartersRandom[1]
                matrizMedias[3][3] = quartersRandom[2]
                matrizMedias[4][3] = max_valueRandom

            #executado no final de cada simulação
            distFinalDRL.append(arrayDist[0])
            distFinalAvoid.append(arrayDist[1])
            distFinalFree.append(arrayDist[2])
            distFinalRandom.append(arrayDist[3])


            if debug:
                print("terminou loop do run. i = %d" % (i))
            # agent.environment.close()
            # agent.environment.close()

            dt = datetime.datetime.now()
            if j == repeticoes -1:
                print("vai gerar a animação")
                FFwriter = animation.FFMpegWriter()
                ani = FuncAnimation(fig, agent.environment.update, interval=10, blit=True, repeat=False,
                                frames=np.linspace(0, agent.environment.FlightModel.duracao, agent.environment.FlightModel.duracao,
                                                   endpoint=False))\
                     .save('results/'+str(dt) + '.gif', writer=FFwriter)

                plt.show()
                plt.clf()
            else:
                print("não vai gerar a animação")
        momentosCriticoRepeticao.append(momentosCriticoEpisodio)
        momentosConflitoRepeticao.append(momentosConflitoEpisodio)

        momentosCriticoRepeticaoAvoid.append(momentosCriticoEpisodioAvoid)
        momentosConflitoRepeticaoAvoid.append(momentosConflitoEpisodioAvoid)

        momentosCriticoRepeticaoFree.append(momentosCriticoEpisodioFree)
        momentosConflitoRepeticaoFree.append(momentosConflitoEpisodioFree)

        momentosCriticoRepeticaoRandom.append(momentosCriticoEpisodioRandom)
        momentosConflitoRepeticaoRandom.append(momentosConflitoEpisodioRandom)

        agent.reset()

        # # # habilitar para ver a evolução do boxplot em uma sessão de treinamento
        # dfSessao = pd.DataFrame(inverteMatriz(distSimulacoes, repeticoes ))
        # # print(dfSessao)
        # dfSessao.boxplot()
        # plt.show()

        mediaDistSimulacoes[k] = distSimulacoes[j]
        mediaDistSimulacoesAvoid[k] = distSimulacoesAvoid[j]
        mediaDistSimulacoesFree[k] = distSimulacoesFree[j]
        mediaDistSimulacoesRandom[k] = distSimulacoesRandom[j]

    # print("momentos criticos DRL")
    # print(momentosCriticoRepeticao)
    # print("momentos conflito DRL")
    # print(momentosConflitoRepeticao)
    #
    # print("momentos criticos Avoid")
    # print(momentosCriticoRepeticaoAvoid)
    # print("momentos conflito Avoid")
    # print(momentosConflitoRepeticaoAvoid)


    print("mediaDistSimulacoes")
    print(mediaDistSimulacoes)
    print("mediaDistSimulacoesAvoid")
    print(mediaDistSimulacoesAvoid)
    print("distFinalDRL:")
    print(distFinalDRL)

    # preencheMatrizMedias(matrizMedias, 0, mediaDistSimulacoes, mediaRepeticoes)
    # preencheMatrizMedias(matrizMedias, 1, mediaDistSimulacoesAvoid, mediaRepeticoes)
    # preencheMatrizMedias(matrizMedias, 2, mediaDistSimulacoesFree, mediaRepeticoes)
    # preencheMatrizMedias(matrizMedias, 3, mediaDistSimulacoesRandom, mediaRepeticoes)

    # Habilitar para que a animação rode ao final do treinamento
    # ani = FuncAnimation(fig, agent.environment.update, interval=10, blit=True, repeat=False,
    #                     frames=np.linspace(0, agent.environment.FlightModel.duracao,
    #                                        agent.environment.FlightModel.duracao,
    #                                        endpoint=False))
    plt.show()

    # plt.savefig('crap.png')

    plt.clf()
    ############################################################
    # Momentos críticos e de conflito
    ############################################################
    ntotalsimulacoes = mediaRepeticoes * repeticoes
    window = round(ntotalsimulacoes / 10)
    print("janela calculada: %d" % (window))

    flat_list = [x for xs in momentosCriticoRepeticao for x in xs]
    plt.yticks(np.arange(0.0, 0.75, step=0.05))  # Set label locations.
    plt.ylim([0, 0.75])
    plt.plot(moving_average(flat_list, n=window), label='momentos Críticos DRL', color="blue")
    # flat_list = [x for xs in momentosConflitoRepeticao for x in xs]
    # plt.plot(moving_average(flat_list, n=window), label='momentos Conflito DRL', color="blue")

    flat_list = [x for xs in momentosCriticoRepeticaoAvoid for x in xs]
    plt.plot(moving_average(flat_list, n=window), label='momentos Críticos Avoid', color="orange")
    # flat_list = [x for xs in momentosConflitoRepeticaoAvoid for x in xs]
    # plt.plot(moving_average(flat_list, n=window), label='momentos Conflito Avoid', color="orange")
    #
    flat_list = [x for xs in momentosCriticoRepeticaoFree for x in xs]
    plt.plot(moving_average(flat_list, n=window), label='momentos Críticos Free', color="red")
    # flat_list = [x for xs in momentosConflitoRepeticaoFree for x in xs]
    # plt.plot(moving_average(flat_list, n=window), label='momentos Conflito Free', color="red")
    #
    flat_list = [x for xs in momentosCriticoRepeticaoRandom for x in xs]
    plt.plot(moving_average(flat_list, n=window), label='momentos Críticos DRL', color="green")
    # flat_list = [x for xs in momentosConflitoRepeticaoRandom for x in xs]
    # plt.plot(moving_average(flat_list, n=window), label='momentos Conflito DRL', color="green")


    plt.savefig('results/momentosConflitoRepeticao' + str(dt) + '.png')
    # plt.show()

    plt.clf()
    ############################################################
    # Boxplot das médias por estratégia
    # ############################################################
    # textstr = '\n'.join((
    #
    #     r'passos por simulação=%d' % (nb_timesteps,),
    #     r'número de simulações=%d' % (repeticoes,),
    #     r'médias após %d simulações:' % (mediaRepeticoes,),
    #     r'Q1 = %.2f mediana = %.2f Q3 = %.2f' % (matrizMedias[1][0],matrizMedias[2][0],matrizMedias[3][0],)
    # ))
    # plt.xlim(0, 5)
    # plt.ylim(-3, 142)
    # # print textstr
    # plt.text(-6, 50, textstr, fontsize=5)
    # plt.grid(True)
    # plt.subplots_adjust(left=0.50)
    #
    # plt.yticks(fontsize=5)
    # plt.xticks(fontsize=5)
    # plt.yticks(np.arange(0, 140, step=5))  # Set label locations.
    #
    # print("quantMomentosConflitoAvoid = %d" %(quantMomentosConflitoAvoid))
    #
    # df1 = pd.DataFrame(matrizMedias, columns=['DRL', 'Avoid', 'Free', 'Random'])
    # print("data frame:")
    # print(df1)
    # df1.boxplot()
    # plt.savefig('results/boxplot Estratégias' + str(dt) + '.png')
    # # plt.show()

    ############################################################
    # Quantidade de momentos críticos
    ############################################################

    # plt.text(1, quantMomentosConflito + 1, quantMomentosConflito, fontsize=10)
    # plt.text(1, quantMomentosMonitor + 1, quantMomentosMonitor, fontsize=10)
    # plt.text(1, quantMomentosCritico + 1, quantMomentosCritico, fontsize=10)
    # plt.plot(1, quantMomentosCritico, color="red", marker='o')
    # plt.plot(1, quantMomentosConflito, color="orange", marker='o')
    # plt.plot(1, quantMomentosMonitor, color="green", marker='o')
    #
    # plt.text(2, quantMomentosConflitoAvoid + 1, quantMomentosConflitoAvoid, fontsize=10)
    # plt.text(2, quantMomentosMonitorAvoid + 1, quantMomentosMonitorAvoid, fontsize=10)
    # plt.plot(2, quantMomentosConflitoAvoid, color="orange", marker='o')
    # plt.plot(2, quantMomentosMonitorAvoid, color="green", marker='o')
    # plt.text(2, quantMomentosCriticoAvoid + 1, quantMomentosCriticoAvoid, fontsize=10)
    # plt.plot(2, quantMomentosCriticoAvoid, color="red", marker='o')
    #
    # plt.text(3, quantMomentosConflitoFree + 1, quantMomentosConflitoFree, fontsize=10)
    # plt.text(3, quantMomentosMonitorFree + 1, quantMomentosMonitorFree, fontsize=10)
    # plt.plot(3, quantMomentosConflitoFree, color="orange", marker='o')
    # plt.plot(3, quantMomentosMonitorFree, color="green", marker='o')
    # plt.text(3, quantMomentosCriticoFree + 1, quantMomentosCriticoFree, fontsize=10)
    # plt.plot(3, quantMomentosCriticoFree, color="red", marker='o')
    #
    # plt.text(4, quantMomentosConflitoRandom + 1, quantMomentosConflitoRandom, fontsize=10)
    # plt.text(4, quantMomentosMonitorRandom + 1, quantMomentosMonitorRandom, fontsize=10)
    # plt.plot(4, quantMomentosConflitoRandom, color="orange", marker='o')
    # plt.plot(4, quantMomentosMonitorRandom, color="green", marker='o')
    # plt.text(4, quantMomentosCriticoRandom + 1, quantMomentosCriticoRandom, fontsize=10)
    # plt.plot(4, quantMomentosCriticoRandom, color="red", marker='o')
    #
    # labels = [' ', 'DRL', 'Avoid', 'Free', 'Random', ' ']
    # x = [0, 1, 2, 3, 4, 5]
    # plt.xticks(x, labels)
    # plt.show()

    plt.clf()
    ############################################################
    # Média móvel da distância final para o destino
    ############################################################

    plt.plot(moving_average(distFinalDRL, n=window), label='DRL')
    plt.plot(moving_average(distFinalAvoid, n=window), label='Avoid')
    plt.plot(moving_average(distFinalRandom, n=window), label='Random')
    # plt.title('Média móvel ao final da simulação. Janela = %d' %(window))
    # textstr = '\n'.join((
    #
    #     r'Média DRL    =%f' % (np.mean(distFinalDRL)),
    #     r'Média Avoid  =%f' % (np.mean(distFinalAvoid)),
    #     r'Média Random =%f' % (np.mean(distFinalRandom)),
    # ))
    # # print textstr
    # plt.text(1, 190, textstr, fontsize=10)
    plt.yticks(np.arange(50, 250, step=10))  # Set label locations.

    # plt.legend()
    # plt.show()
    plt.savefig('results/distFinal' + str(dt) + '.png')

    #salvar o modelo para utilizar posteriormente
    agent.save('model/'+str(dt))

    # print("arquitetura da rede:")
    # print(agent.get_architecture())

    agent.environment.close()
