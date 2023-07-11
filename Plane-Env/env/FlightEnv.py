import numpy as np
from tensorforce.environments import Environment
from .FlightModel import FlightModel



class PlaneEnvironment(Environment):

    def update(self, momento):
        print("momento =%d" % (momento))

        # #atualizar o plot da aeronave zero
        # if momento > 0:
        #     self.FlightModel.traceAllx[0].appendleft(self.FlightModel.historicoDeVoos[int(momento - 1)][0][0])
        #     self.FlightModel.traceAlly[0].appendleft(self.FlightModel.historicoDeVoos[int(momento - 1)][0][1])
        # proxPosicao = self.FlightModel.historicoDeVoos[int(momento)][0]
        # self.FlightModel.plotsAcft[0].set_data(proxPosicao)
        # self.FlightModel.traceAll[0].set_data(self.FlightModel.traceAllx[0], self.FlightModel.traceAlly[0])

        # atualizar os plots de todas as aeronaves
        for i in range( self.FlightModel.quantAeronaves):
            # print("loop do update i =%d " % (i))
            # print(self.FlightModel.traceAllx[i])
            if momento > 0:
                self.FlightModel.traceAllx[i].appendleft(self.FlightModel.historicoDeVoos[int(momento - 1)][i][0])
                self.FlightModel.traceAlly[i].appendleft(self.FlightModel.historicoDeVoos[int(momento - 1)][i][1])

            # print(self.FlightModel.historicoDeVoos[int(momento)][i])
            proxPosicao = self.FlightModel.historicoDeVoos[int(momento)][i]
            # movimenta(momento, voos[i])
            # print(proxPosicao)
            self.FlightModel.plotsAcft[i].set_data(proxPosicao)
            self.FlightModel.traceAll[i].set_data(self.FlightModel.traceAllx[i], self.FlightModel.traceAlly[i])

            # se houver um conflito para a aeronave em i, plotar um círculo vermelho ao redor da pista dela
            if (self.FlightModel.histConflitos[int(momento)] is not None and i in self.FlightModel.histConflitos[int(momento)]):
                print("momento = %d vai plotar os conflitos" % (momento))
                print(self.FlightModel.histConflitos[int(momento)])
                print("plotando conflito da aeronave=%d" % (i))
                print(self.FlightModel.historicoDeVoos[int(momento)][i])
                self.FlightModel.circuloConflito[i].set_data(self.FlightModel.historicoDeVoos[int(momento)][i])
                if self.FlightModel.voos[0][i].desvio is True:
                    self.FlightModel.circuloDesvio[i].set_data(self.FlightModel.historicoDeVoos[int(momento)][i])
                    self.FlightModel.circuloConflito[i].set_data([[], []])
            else:  # limpar a marca do conflito que havia sido plotada
                self.FlightModel.circuloConflito[i].set_data([[], []])
                self.FlightModel.circuloDesvio[i].set_data([[], []])

        varDeRetorno = self.FlightModel.plotsAcft
        varDeRetorno = np.append(varDeRetorno, self.FlightModel.traceAll)
        varDeRetorno = np.append(varDeRetorno, self.FlightModel.circuloConflito)
        varDeRetorno = np.append(varDeRetorno, self.FlightModel.circuloDesvio)

        # if momento == 0:
        #     self.FlightModel.history_x.clear()
        #     self.FlightModel.history_y.clear()
        #     for i in range(self.FlightModel.quantAeronaves):
        #         self.FlightModel.traceAllx[i].clear()
        #         self.FlightModel.traceAlly[i].clear()
        # self.FlightModel.history_x.appendleft(momento - 1)
        # self.FlightModel.history_y.appendleft(momento - 1)

        # trace.set_data(history_x, history_y)
        # varDeRetorno = np.append(varDeRetorno, trace)

        # plotZero.set_data(momento / 2, momento + 500)
        # varDeRetorno = np.append(varDeRetorno, plotZero)

        # acft.set_data(momento, momento)
        # varDeRetorno = np.append(varDeRetorno,acft)

        # print (varDeRetorno)
        return varDeRetorno

    def __init__(self, n_episodes, max_step_per_episode, ax, repeticoes, repeticaoAtual, altura, largura, quantAeronaves,
                 quantVertiports, toleranciaLinear, nnearest, valendo, debug = True):
        super().__init__()

        self.max_step_per_episode = max_step_per_episode
        self.n_episodes = n_episodes
        self.FlightModel = FlightModel(self.n_episodes, ax, repeticoes, repeticaoAtual, debug, valendo, altura, largura, quantAeronaves,
                 quantVertiports, toleranciaLinear, nnearest)
        self.NUM_ACTIONS = len(self.FlightModel.action_vec)
        self.finished = False
        self.episode_end = False
        self.STATES_SIZE = len(self.FlightModel.obs)
        # print("definiu states size como: %f" %(self.STATES_SIZE))
        self.nnearest = nnearest


    def states(self):
        return dict(
            type="int", shape=self.STATES_SIZE , num_values = 2000
            # intrusoDist=dict(type='int', shape=(1), num_values=142),
            # intrusoAngulo=dict(type='int', shape=(1), num_values=36),
            # anguloInterceptacao=dict(type='int', shape=(1), num_values=36),
            # proaDestino=dict(type='int', shape=(1), num_values=36)
        )


    def actions(self):
        return {
            # "thrust": dict(type="int", num_values=6),
            "desvio": dict(type="int", num_values=19)
        }

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return self.max_step_per_episode

    # Optional
    def close(self):
        super().close()

    def reset(self, n_episodes, ax, repeticoes, k, debug):
        state = np.zeros(shape=(self.STATES_SIZE,))
        self.FlightModel = FlightModel(self.n_episodes, ax, repeticoes, k, debug, True)
        return state

    def execute(self, current_episode, debug, ax, duracao, actions):
        if debug:
            print("actions que recebeu no execute:")
            print(actions)
        reward = 0
        # for i in range(1, self.n_episodes + 1):
        #     print("execute do flightEnv. i = %d. " %(i))
        next_state, reward, distAvoid, distFree, distRandom, arrayDist = self.FlightModel.compute_timestep(actions, current_episode, debug, duracao,self.nnearest)
        # reward += self.reward()
        # # if self.terminal():
        # #     print("saindo do loop porque terminal retornou verdadeiro")
        # #     reward = reward / current_episode
        # #     break
        #
        # if current_episode == self.n_episodes:
        #     reward = reward / self.n_episodes
        # reward = self.reward()
        if debug:
            print("reward antes de retornar do execute: %f" %(reward))
            print("states:")
            print(next_state)
        # print("vai chamar a função terminal")
        terminal = self.terminal()
        return next_state, terminal, reward, distAvoid, distFree, distRandom, arrayDist

    def terminal(self):
        # return False
        # print("função terminal. valor de timestep = %d" %(self.FlightModel.timestep))
        # print("valor de max_step_per_episode")
        # print(self.max_step_per_episode)
        self.finished = self.FlightModel.DistDestino < 1  # chegou até o destino
        #TODO colocar o episode_end como uma variável
        # self.episode_end = (self.FlightModel.timestep >= self.max_step_per_episode)
        self.episode_end = (self.FlightModel.timestep >= self.n_episodes)
        # print("função terminal. episode_end:")
        # print(self.episode_end)
        return self.finished or self.episode_end

    # def reward(self):
    #     if self.finished:
    #         reward = np.log(self.FlightModel.DistDestino)
    #     else:
    #         reward = -1
    #     return reward
