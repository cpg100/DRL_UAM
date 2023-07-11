import math
from math import cos, sin, ceil, floor
import numpy as np
from random import randrange
import random as rand
import matplotlib.pyplot as plt
from collections import deque
from pprint import pprint
from numpy import arcsin
from numpy.linalg import norm
# from graph_utils import plot_duo, plot_multiple, plot_xy

# from .AnimatePlane import animate_plane
# from ..utils import write_to_txt

# from .test_animation import animate_plane


class Voo:
    def __init__(self, numero, origem, destino, vertis, duracao, debug):
        self.numero = numero #número do voo
        self.origem = origem #código numérico que indica o vertiport de origem
        self.destino = destino
        self.decolagem = 0
        distX = vertis[0][destino][0] - vertis[0][origem][0]
        distY = vertis[0][destino][1] - vertis[0][origem][1]
        hipotenusa = math.sqrt(distX ** 2 + distY ** 2)
        self.velocidade = hipotenusa / (duracao)
        self.velocidadeInicial = hipotenusa / (duracao)
        self.vetorX = distX/hipotenusa
        self.vetorY = distY/hipotenusa
        if debug:
            print("origem x = %f, destino x = %f, vetor x = %f" % (
                vertis[0][origem][0], vertis[0][destino][0], self.vetorX))
            print("origem y = %f, destino y = %f, vetor y = %f" % (vertis[0][origem][1], vertis[0][destino][1], self.vetorY))

        self.vetorXRandom = self.vetorX
        self.vetorYRandom = self.vetorY

        self.vetorXAvoid = self.vetorX
        self.vetorYAvoid = self.vetorY
        if debug:
            print("vetor xAvoid = %f"%(self.vetorXAvoid))
            print("vetor yAvoid = %f"%(self.vetorYAvoid))
        self.vetorXFree = self.vetorX
        self.vetorYFree = self.vetorY

        self.xAtual = vertis[0][origem][0]
        self.yAtual = vertis[0][origem][1]
        self.xAtualFree = self.xAtual #posições auxiliares para indicar onde a aeronave estaria por outra estratégia
        self.yAtualFree = self.yAtual #posições auxiliares para indicar onde a aeronave estaria por outra estratégia
        self.xAtualRandom = self.xAtual  # posições auxiliares para indicar onde a aeronave estaria por outra estratégia
        self.yAtualRandom = self.yAtual  # posições auxiliares para indicar onde a aeronave estaria por outra estratégia
        self.xAtualAvoid = self.xAtual  # posições auxiliares para indicar onde a aeronave estaria por outra estratégia
        self.yAtualAvoid = self.yAtual  # posições auxiliares para indicar onde a aeronave estaria por outra estratégia



        self.desvio = False
        self.desvioGraus = 0
        self.desvioDist = 0


class FlightModel:



    def __init__(self, duracao, ax,repeticoes, repeticaoAtual, debug, valendo, altura, largura, quantAeronaves,
                 quantVertiports, toleranciaLinear, nnearest):

        if debug:
            print("início da função de inicialização do flight model. valendo = ")
            print(valendo)

        self.action_vec = [
            [desvio] for desvio in range(-9, 9)
        ]

        """
                    OBSERVATIONS
                    States vec for RL stocking position and velocity
                    """

        self.DistDestino = 6546756  # float ('inf') #distância para o vertiport de destino. o valor é inicializado com +infinito
        self.intrusoDist = 0  # float('inf') #distância para a aeronave em conflito. o valor é inicializado com +infinito
        self.intrusoDistAnterior = 0 # será usado para medir a variação de distância para o intruso
        self.intrusoAngulo = 0  # angulo do intruso em relação à minha aeronave 12h = 0 graus, 3h = 45 graus
        self.anguloInterceptacao = 0  # angulo entre a trajetória do intruso e a minha aeronave.
        proaDestino = 0
        # em outras palavras, quantos graus a aeronave precisa curvar para manter uma trajetória paralela à minha

        # variáveis do state:
        # - distância do intruso
        # - ângulo formado com a minha trajetória
        # - proa direta para o intruso
        self.nnearest = nnearest
        self.dataMatrix = np.zeros((3 * nnearest + 0,), dtype=int)
        self.dataMatrixAnterior = np.zeros((3 * nnearest + 0,), dtype=int) #para guardar os valores entre as iterações
        if debug:
            print("data matrix definida no início do flight model:")
            print(self.dataMatrix)
        self.obs = [0]
        self.obs = self.obs + self.dataMatrix
        if debug:
            print("self.obs definida no início do flight model:")
            print(self.obs)


        # limites do espaço aéreo que será utilizado na simulação
        if valendo:
            self.altura = altura
            self.largura = largura
            self.quantAeronaves = quantAeronaves
            self.quantVertiports = quantVertiports
            self.toleranciaLinear = toleranciaLinear  # metros distância que será usada para determinar se há conflito
            self.duracao = duracao
            # self.duracao = 300  # segundos

            self.vertis = self.geraVertiports(self.quantVertiports, debug)
            self.historicoDeVoos = np.empty(self.duracao+1, dtype=object)
            self.voos = self.geraVoos(self.quantAeronaves, self.quantVertiports, debug)
            self.statDistConflitos = []
            self.proaRadianosZero = 0

            # print("voos gerados:")
            # print(self.voos)
            # print("dados do voo 0. origem x =")
            # pprint(vars(self.voos[0][0]))




            self.plotsAcft = np.empty(self.quantAeronaves, dtype=object)
            self.traceAll = np.empty(self.quantAeronaves, dtype=object)
            self.circuloConflito = np.empty(self.quantAeronaves, dtype=object)
            self.circuloDesvio = np.empty(self.quantAeronaves, dtype=object)
            self.histConflitos = np.empty(self.duracao+1, dtype=object)  # será usado para indicar as aeronaves em conflito
            self.histConflitosAvoid = np.empty(self.duracao + 1,
                                          dtype=object)  # serão usados arrays diferentes para cada estratégia
            self.histConflitosFree = np.empty(self.duracao + 1,
                                               dtype=object)  # serão usados arrays diferentes para cada estratégia
            self.histConflitosRandom = np.empty(self.duracao + 1,
                                               dtype=object)  # serão usados arrays diferentes para cada estratégia

            if repeticaoAtual == repeticoes-1:
                print("vai entrar no loop de inicialização do ax.plot")
                for i in range(self.quantAeronaves):
                    origem = self.voos[0][i].origem
                    print("origem do voo=%d" % (origem))
                    print(self.vertis[0][origem])
                    destino = self.voos[0][i].destino
                    print("i = %d. plotando decolagem de %f, %f" %(i, self.vertis[0][origem][0], self.vertis[0][origem][1]))
                    self.plotsAcft[i], = ax.plot(self.vertis[0][origem][0], self.vertis[0][origem][1], "r+")
                    self.traceAll[i], = ax.plot([], [], '.-', lw=1, ms=2)
                    self.circuloConflito[i], = ax.plot([], [], 'ro')
                    self.circuloDesvio[i], = ax.plot([], [], 'go')

                #o plot da aeronave zero será diferenciado
                self.plotsAcft[0], = ax.plot(self.vertis[0][origem][0], self.vertis[0][origem][1], "bs")


                for i in range(self.quantVertiports):
                    temp = plt.Circle((self.vertis[0][i][0], self.vertis[0][i][1]), 10.1, color='b', fill=False)
                    ax.add_patch(temp)
                ax.axis("equal")

                plt.rcParams["figure.figsize"] = 4, 3
                history_len = 300  # how many trajectory points to display
                history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

                self.traceAllx = np.empty(self.quantAeronaves, dtype=object)
                self.traceAlly = np.empty(self.quantAeronaves, dtype=object)
                for i in range(self.quantAeronaves):
                    self.traceAllx[i], self.traceAlly[i] = deque(maxlen=history_len), deque(maxlen=history_len)








            """
            ACTIONS:
            Action vec for RL stocking thrust and theta values
            """
            self.timestep = 0  # init timestep
            self.timestep_max = duracao  # Max number of timestep per episode



    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def isNaN(self, num):
        return num != num

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        # print("v1_u:")
        # print(v1_u)
        v2_u = self.unit_vector(v2)
        # print("v2_u:")
        # print(v2_u)
        temp = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if math.isnan(temp):
            # print("retornando zero porque temp=nan")
            return 0
        else:
            # print("a função auxiliar garantiu que temp NÃO é nan")
            return temp

    def calculaDistVertiports(self,origem, destino, debug):
        xOrigem, yOrigem = self.vertis[0][origem][0],self.vertis[0][origem][1]
        xDestino, yDestino = self.vertis[0][destino][0], self.vertis[0][destino][1]
        deltaX = xDestino-xOrigem
        deltaY = yDestino-yOrigem
        if debug:
            print("delta x = %f, delta y = %f" %(deltaX, deltaY))
        return math.sqrt(deltaX**2 + deltaY**2)

    def geraVoos (self, quantAeronaves, quantVertiports, debug):
        if debug:
            print("Função geraVoos")
        voos = np.empty(quantAeronaves, dtype=object)
        posicaoTempVoos = np.empty(self.quantAeronaves, dtype=object)
        for i in range(quantAeronaves):
            if debug:
                print("criando voo i=%d" % (i))
            origem = randrange(quantVertiports)

            # escolher o destino diferente da origem
            numbers = list(range(0, quantVertiports))
            numbers.remove(origem)
            #garantir que exista uma distância mínima até o destino para que os desvios possam ser executados
            destino = rand.choice(numbers)
            distDestino = self.calculaDistVertiports(origem, destino, debug)
            while distDestino < 300:
                destino = rand.choice(numbers)
                distDestino = self.calculaDistVertiports(origem, destino, debug)
            if debug:
                print("origem = %d" %(origem,))
                print("destino = %d" % (destino,))
            posicaoTempVoos[i] = (self.vertis[0][origem][0],self.vertis[0][origem][1])

            temp = Voo(i, origem, destino, self.vertis, self.duracao, debug)
            voos[i] = temp
        self.historicoDeVoos[0] = posicaoTempVoos
        return voos,

    def geraVertiports(self, quantVertiports, debug):
        vertis = np.empty(quantVertiports, dtype=object)
        for v in range(quantVertiports):
            vertis[v] = [rand.uniform(0, self.largura), rand.uniform(0, self.altura)]
            # if debug:
            #     print("v= %d, vertX=%f, vertY=%f" %(v, vertis[v][0], vertis[v][1]))
            #     print(vertis[v])
        return vertis,

    # acrescentar a movimentação quando está ocorrendo desvio
    def movimenta(self, momento, voo, vertis):
        # print("função movimenta antes de executar. xatual = %f, vetor x = %f velocidade = %.10f" % (voo.xAtual, voo.vetorX, voo.velocidade))
        # print("função movimenta antes de executar. xatual = %f, vetor x = %f velocidade = %.10f - avoid" % (voo.xAtualAvoid, voo.vetorXAvoid, voo.velocidade))
        # print("função movimenta antes de executar. xatual = %f, vetor x = %f velocidade = %.10f - free" % (voo.xAtualFree, voo.vetorXFree, voo.velocidade))
        # print("função movimenta antes de executar. xatual = %f, vetor x = %f velocidade = %.10f - random"
        #       % (voo.xAtualRandom, voo.vetorXRandom, voo.velocidade))
        voo.xAtual = voo.xAtual + voo.vetorX * voo.velocidade
        voo.yAtual = voo.yAtual + voo.vetorY * voo.velocidade

        voo.xAtualAvoid = voo.xAtualAvoid + voo.vetorXAvoid * voo.velocidade
        voo.yAtualAvoid = voo.yAtualAvoid + voo.vetorYAvoid * voo.velocidade

        voo.xAtualRandom = voo.xAtualRandom + voo.vetorXRandom * voo.velocidade
        voo.yAtualRandom = voo.yAtualRandom + voo.vetorYRandom * voo.velocidade

        voo.xAtualFree = voo.xAtualFree + voo.vetorXFree * voo.velocidade
        voo.yAtualFree = voo.yAtualFree + voo.vetorYFree * voo.velocidade

        # print("função movimenta. xatual = %f, vetor x = %f velocidade = %f" % (voo.xAtual, voo.vetorX, voo.velocidade))
        # print("função movimenta. xatual = %f, vetor x = %f velocidade = %f - avoid" % (
        # voo.xAtualAvoid, voo.vetorXAvoid, voo.velocidade))
        # print("função movimenta. xatual = %f, vetor x = %f velocidade = %f - free" % (
        # voo.xAtualFree, voo.vetorXFree, voo.velocidade))
        # print("função movimenta. xatual = %f, vetor x = %f velocidade = %f - random" % (
        #     voo.xAtualRandom, voo.vetorXRandom, voo.velocidade))

        return [[voo.xAtual, voo.yAtual],[voo.xAtualAvoid,voo.yAtualAvoid],[voo.xAtualFree,voo.yAtualFree],[voo.xAtualRandom,voo.yAtualRandom]];

    #função utilizada para definir como será ordenada uma lista
    def ordemLista(self, chaveValor):
        return chaveValor[1]

    def detectaConflitoPorEstrategia(self, arrayHistConflitos, current_timestep, posEstrategia,
                                      posicaoTempVoos, debug, k, nnearest ):
        if debug:
            print("posEstrategia antes de detectar o conflito:")
            print(posEstrategia)

        distX = abs(posEstrategia[0] - posicaoTempVoos[k][0])
        distY = abs(posEstrategia[1] - posicaoTempVoos[k][1])

        if distX < self.toleranciaLinear and distY < self.toleranciaLinear:
            if debug:
                print("entrou no if porque distX = %f e distY =%f" % (distX, distY))
                print("lista de conflitos antes do if")
                print(arrayHistConflitos[current_timestep])

            # verificar se a separação atual é menor do que a anterior
            if len(arrayHistConflitos[current_timestep]) > 0:
                distanciaAcftAtual = math.sqrt(distX ** 2 + distY ** 2)

                # calcular a distância para o conflito que já está registrado
                secondAcft = arrayHistConflitos[current_timestep][1]
                distX2nd = abs(posEstrategia[0] - posicaoTempVoos[secondAcft][0])
                distY2nd = abs(posEstrategia[1] - posicaoTempVoos[secondAcft][1])
                if debug:
                    print("distX2nd = %f, distY2nd = %f" % (distX2nd, distY2nd))
                distanciaSecond = math.sqrt(distX2nd ** 2 + distY2nd ** 2)
                if distanciaAcftAtual < distanciaSecond:
                    if debug:
                        print ("Adicionando o conflito com a aeronave k = %d." % k)
                    arrayHistConflitos[current_timestep].insert(1, k)
                    if debug:
                        print("Array de conflitos após adicionar o novo conflito: k = %d.")
                        print(arrayHistConflitos[current_timestep])
                    if len(arrayHistConflitos[current_timestep]) > self.nnearest+1:
                        if debug:
                            print("Número de conflitos ultrapassou nnearest. vai apagar os últimos elementos. resultado após apagar:")
                        del arrayHistConflitos[current_timestep][self.nnearest+1:]
                        if debug:
                            print(arrayHistConflitos[current_timestep])
                # else:
                #     if debug:
                #         print("distância previamente calculada de %f ainda é menor" %(distanciaAcftAtual))

            else:
                # neste caso, não há um conflito anterior. o conflito atual é adicionado
                arrayHistConflitos[current_timestep] = [0, k]
                if debug:
                    print("inserindo conflito pois não havia outro. histórico de conflitos após inserir o voo k e o pivot:")
                    print(arrayHistConflitos[current_timestep])

            # habilitar o else para detectar mais de um conflito
            # else:
            #     histConflitos[i].append(xOrdenado[k][0])
            #     histConflitos[i].append(xOrdenado[pivot][0])

        else:  #
            if debug:
                print("não entrou no if porque distX = %f e distY =%f" % (distX, distY))



    def calculaDistanciaPorEstrategia(self, arrayConflitos, debug, posEstrategia, posicaoTempVoos):
        # se houver conflito, calcular os valores da observação e o reward
        if debug:
            print("vai fazer os cálculos para o seguinte conflito:")
            print(arrayConflitos)
            print("posEstrategia recebido:")
            print(posEstrategia)
        secondAcft = int(arrayConflitos[1])
        if debug:
            print("secondAcft = %d" % (secondAcft))

        # cálculo da distância
        if debug:
            print("x aeronave zero: %f, x segunda aeronave: %f" % (
                posEstrategia[0], posicaoTempVoos[secondAcft][0]))
            print("y aeronave zero: %f, y segunda aeronave: %f" % (
                posEstrategia[1], posicaoTempVoos[secondAcft][1]))
        distX = abs(posEstrategia[0] - posicaoTempVoos[secondAcft][0])
        distY = abs(posEstrategia[1] - posicaoTempVoos[secondAcft][1])
        if debug:
            print("distX = %f, distY = %f" % (distX, distY))
        intrusoDist = math.sqrt(distX ** 2 + distY ** 2)
        if debug:
            print("distância calculada: %f" % (intrusoDist))
            if(intrusoDist > 200):
                print("distância anômala")

        return intrusoDist

    # def ajustarVelocidadeDRL(self, debug,vooZero, proaRadianos, action):
    #     nAction = action['desvio']
    #     if debug:
    #         print("vai ajustar velocidade de acordo com action = %d" %(nAction))
    #         print("velocidade antes de ajustar %f" %(vooZero.velocidade))
    #     vooZero.velocidade = vooZero.velocidadeInicial * nAction/19
    #     if debug:
    #         print("nova velocidade: %f" %(vooZero.velocidade))


    def executarDesvioDRL(self, debug,vooZero, proaRadianos, action):
        if debug:
            print(
                "calculando o desvio do voo zero. vetor xAtual = %f, yAtual = %f, proa Radianos = %f. equivalente a = %f graus" % (
                    vooZero.vetorX, vooZero.vetorY, proaRadianos, math.degrees(proaRadianos)))
            print(vooZero.desvio)

        # converter a action em uma proa
        # 0 = 90 graus à esquerda, 9 = manter proa, 18 = 90 graus à direita
        if debug:
            print("valor do action = %d " % (action['desvio']))
        desvioGraus = (action['desvio'] - 9) * 10
        if debug:
            print("nova proa = %d " % (desvioGraus))
            print("nova proa em graus = %f" % (math.degrees(proaRadianos) - desvioGraus))
        novaProa = math.radians(math.degrees(proaRadianos) - desvioGraus)
        if debug:
            print("nova proa calculada %f, em radianos %f" % (math.degrees(novaProa), novaProa))
        moduloVetor = math.sqrt(vooZero.vetorX ** 2 + vooZero.vetorY ** 2)
        if debug:
            print("novo módulo do vetor para o voo zero: %f" %(moduloVetor))
        vooZero.vetorX = math.cos(novaProa)
        vooZero.vetorY = math.sin(novaProa)

    def executarDesvioAvoid(self, debug, secondAcft, vooZero):

        vooConflito = self.voos[0][secondAcft]
        vetorXIntruso = -1*(vooConflito.xAtual - vooZero.xAtualAvoid) #a multiplicação por -1 visa já obter a proa oposta
        vetorYIntruso = -1*(vooConflito.yAtual - vooZero.yAtualAvoid) #a multiplicação por -1 visa já obter a proa oposta
        proaFugirIntrusoRadianos = math.atan2(vetorYIntruso, vetorXIntruso)

        if debug:
            print("vetorXIntruso = %f, vetorYIntruso = %f" %(vetorXIntruso,vetorYIntruso))
            print("vai definir a proa avoid. proa para fugir do intruso: %f " % ( math.degrees(proaFugirIntrusoRadianos)))

        hipotenusa = math.sqrt(vetorXIntruso ** 2 + vetorYIntruso ** 2)
        if hipotenusa < 0.000000000000000000001:
            hipotenusa = 0.000000000000000000001
        vooZero.vetorXAvoid = vetorXIntruso / hipotenusa
        vooZero.vetorYAvoid = vetorYIntruso / hipotenusa

    def executarDesvioRandom(self,vooZero,debug):
        desvioRandom = randrange(19)
        desvioRandomGraus = (desvioRandom - 9) * 10
        # if debug:
        #     print("nova proa = %d " % (desvioGraus))
        #     print("nova proa em graus = %f" % (math.degrees(proaRadianos) - desvioGraus))
        proaRadianosRandom = math.atan2(vooZero.vetorYRandom, vooZero.vetorXRandom)
        novaProaRandom = math.radians(math.degrees(proaRadianosRandom) - desvioRandomGraus)
        if debug:
            print("novaProaRandom calculada %f, em radianos %f" % (math.degrees(novaProaRandom), novaProaRandom))
        moduloVetorRandom = math.sqrt(vooZero.vetorXRandom ** 2 + vooZero.vetorYRandom ** 2)
        vooZero.vetorXRandom = math.cos(novaProaRandom)
        vooZero.vetorYRandom = math.sin(novaProaRandom)

    #posição relativa = proa direta para o intruso
    def calcularAnguloIntruso(self, secondAcft, debug, xAtual, yAtual, vetorX, vetorY):
        #usa os dados da secondAcft para chamar a função auxiliar que irá calcular a proa

        vetorXIntruso = secondAcft.xAtual - xAtual
        vetorYIntruso = secondAcft.yAtual - yAtual

        return self.calcularAnguloIntrusoAux(vetorYIntruso, vetorXIntruso, debug, vetorX, vetorY)

    # posição relativa = proa direta para o intruso
    def calcularAnguloIntrusoAux(self, vetorYIntruso, vetorXIntruso, debug, vetorX, vetorY):
        proaRadianosPintruso = math.atan2(vetorYIntruso, vetorXIntruso)
        if debug:
            print("proa direta para o intruso em radianos = %f. em graus = %f." %
                  (proaRadianosPintruso, math.degrees(proaRadianosPintruso)))

        v1 = (vetorY, vetorX, 0)
        v2 = (vetorYIntruso, vetorXIntruso, 0)
        # anguloIntruso = self.angle_between(v1, v2)
        proaRadianosZero = math.atan2(vetorY, vetorX)
        anguloIntruso = self.anguloEntre(proaRadianosZero, proaRadianosPintruso, debug)

        if debug:
            print("calculando a proa para o intruso. vetor vetorXIntruso = %f, vetorYIntruso = %f , vetorX=%f, vetorY=%f" % (
                vetorXIntruso, vetorYIntruso, vetorX, vetorY))
        # arredondar para o múltiplo de 10 mais próximo
        anguloFinal = round(math.degrees(anguloIntruso) / 10)
        if debug:
            print("angulo para aproar o intruso em radianos = %f. em graus = %f. angulo final = %d" % (
                anguloIntruso, math.degrees(anguloIntruso), anguloFinal))
        return anguloFinal

    def anguloEntre(self, proaRadianosZero, proaRadianos2, debug):
        if math.degrees(proaRadianosZero) < 0:
            proaRadianosZero = proaRadianosZero + 2*np.pi
            if debug:
                print(" inverteu o sinal da proa do voo zero. novo valor = %f" %(math.degrees(proaRadianosZero)))

        if math.degrees(proaRadianos2) < 0:
            proaRadianos2 = proaRadianos2 + 2*np.pi
            if debug:
                print(" inverteu o sinal da proa do voo 2. novo valor = %f" %(math.degrees(proaRadianos2)))

        anguloEntre = proaRadianosZero - proaRadianos2
        if debug:
            print("angulo calculado inicialmente: %f " % (math.degrees(anguloEntre)))
        if anguloEntre < 0:
            anguloEntre = anguloEntre + 2 * np.pi
            if debug:
                print(" inverteu o sinal do anguloEntre. novo valor= %f" % (math.degrees(anguloEntre)))
        return anguloEntre


    # proa relativa = diferença entre a minha proa e a do intruso
    def calculaAnguloInterceptacao(self, vetorX, vetorY, secondAcft, debug):






        self.proaRadianosZero = math.atan2(vetorY, vetorX)
        if debug:
            print("calculando a proa do voo 0. vetor xAtual = %f, yAtual = %f, atan = %f. . = %f graus" % (
                vetorX, vetorY, self.proaRadianosZero, math.degrees(self.proaRadianosZero)))

        proaRadianos2 = math.atan2(secondAcft.vetorY, secondAcft.vetorX)
        if debug:
            print(
                "calculando a proa do voo secondAcft. vetor xAtual = %f, yAtual = %f, atan = %f. . = %f graus" % (
                    secondAcft.vetorX, secondAcft.vetorY, proaRadianos2, math.degrees(proaRadianos2)))

        v1 = (vetorY, vetorX, 0)
        v2 = (secondAcft.vetorY, secondAcft.vetorX, 0)
        anguloCalculado = self.angle_between(v1, v2)
        # interpretar o sinal do ângulo calculado
        # angulo entre os 2 voos
        anguloEntre = self.anguloEntre(self.proaRadianosZero, proaRadianos2, debug)


        anguloFinalConflito = round(math.degrees(anguloEntre) / 10)
        if debug:
            print("angulo calculado em radianos = %f. em graus = %f, angulo entre as proas das acft/10: %d" % (
                anguloEntre, math.degrees(anguloEntre), anguloFinalConflito))
        return anguloFinalConflito
        # self.anguloInterceptacao = anguloFinalConflito

    def calculaReward(self, current_timestep, desvio, debug, vooZero):
        reward = 0
        componenteConflito = 0
        componenteDistDestino =0
        if current_timestep > 0 and self.histConflitos[current_timestep - 1] is not None:
            # calcular o reward
            if self.intrusoDist < self.toleranciaLinear * 0.5:  # distância menor do que o aceitável
                # if self.intrusoDist < 0.00000001:  # para evitar divisão por zero
                #     self.intrusoDist = 0.00000001
                componenteConflito -= 5
                if self.intrusoDist < self.toleranciaLinear * 0.25:  # o intruso entrou na zona crítica
                    componenteConflito -= 5
            else:
                componenteConflito += 1

        else:
            componenteConflito +=10

        # else:
        #     # neste caso em que não há conflito, o reward será negativo caso a IA determine um desvio
        #     if desvio != 9:
        #         reward = -1
        #     else:
        #         reward = 1

        # # teste de reward baseado apenas na distância para o intruso
        # if current_timestep > 0 and self.histConflitos[current_timestep - 1] is not None:
        #     # calcular o reward
        #     if self.intrusoDist < self.intrusoDistAnterior:
        #         reward +=  self.intrusoDist - self.intrusoDistAnterior #a recompensa será negativa e proporcional à diminuição
        #         # if self.intrusoDist < self.toleranciaLinear * 0.25:  # distância menor do que o aceitável
        #         #     if self.intrusoDist < 0.00000001:  # para evitar divisão por zero
        #         #         self.intrusoDist = 0.00000001
        #         #     reward -= 1 / (self.intrusoDist / self.toleranciaLinear)
        #     else:
        #         reward += 1
        # else:
        #     reward += 2

        # else:
        #     # neste caso em que não há conflito, o reward será negativo caso a IA determine um desvio
        #     if desvio != 9:
        #         reward += -1
        #     else:
        #         reward += 1


        # #adicionar uma recompensa positiva para a escolha da proa mais próxima para o destino. Independente da existência de conflito
        xDestino = self.vertis[0][vooZero.destino][0]
        yDestino = self.vertis[0][vooZero.destino][1]

        vetorXIntruso = xDestino - vooZero.xAtual
        vetorYIntruso = yDestino - vooZero.yAtual
        if debug:
            print("vai calcular a posição relativa do destino para definir a recompensa")
        proaDestino = self.calcularAnguloIntrusoAux(vetorYIntruso, vetorXIntruso, debug, vooZero.vetorX, vooZero.vetorY)
        # if debug:
        #     print("proaDestino calculada para o reward: %f" %(proaDestino))
        # #definir uma função que seja negativa à medida que proaDestino se afasta de zero
        # if proaDestino == 0:
        #     if debug:
        #         print("incrementando o reward porque a proa está apontando para o destino")
        #     componenteDistDestino+= 1
        # else:
        #     if debug:
        #         print("decrementando o reward porque a proa está se afastando do destino")
        #     #prevenção para o caso em que o destino está à esquerda (ex. proa relativa 350)
        #     if proaDestino > 18:
        #         if debug:
        #             print("proa destino está à esquerda. vai subtrair %d do valor atual" %(36 - proaDestino))
        #         componenteDistDestino -= 36 - proaDestino
        #     else: #
        #         componenteDistDestino-= proaDestino
        # if debug:
        #     print( "retornando reward %f" % (reward))

        # calcular a média ponderada dos componenentes
        pesoConflito = 1
        reward = componenteConflito + componenteDistDestino
        return reward, proaDestino

    def calculaDistPDestino(self, xAtual, yAtual, xDest, yDest, debug):
        if debug:
            print("vai calcular a distância para o destino. xAtual = %f, xDest = %f,  yAtual = %f, yDest = %f" %(xAtual, xDest, yAtual, yDest))
        dist = math.sqrt((xAtual- xDest)**2 + (yAtual-yDest)**2)
        return dist

    def calculaDataMatrix(self, histConflitos, current_timestep, debug, posDRL, posicaoTempVoos, vetorX,
                                             vetorY, xAtual, yAtual, nnearest):
        matriz = []

        # criar cópia local para não alterar a lista original
        # histConflitosLocal = []
        histConflitosLocal = np.copy(histConflitos[current_timestep])

        if debug:
            print("histórico de conflitos original:")
            print(histConflitos[current_timestep])
            print("cópia local:")
            print(histConflitosLocal)

        #adicionar elementos à lista de conflitos para preencher os nnearest

        while len(histConflitosLocal) > 1:
            # preencher o data matrix com os dados das n aeronaves mais próximas


            secondAcft = self.voos[0][int(histConflitosLocal[1])]
            if debug:
                print("vai calcular as distâncias para DRL")
            self.intrusoDist = int(self.calculaDistanciaPorEstrategia(histConflitosLocal, debug, posDRL,
                                                                      posicaoTempVoos))
            # tentativa de limitar a distância
            if self.intrusoDist > 200:
                self.intrusoDist = 200

            matriz.append(self.intrusoDist)
            # adicionar os valores de distância ao array para calcular a estatística
            self.statDistConflitos.append(self.intrusoDist)

            if debug:
                print("funcâo dataMatrix. distância calculada: %f" % (self.intrusoDist))

            # cálculo do ângulo de interceptação
            anguloInterceptacao = self.calculaAnguloInterceptacao(vetorX, vetorY, secondAcft, debug)
            matriz.append(anguloInterceptacao)

            # calculo do angulo do intruso em relação à minha aeronave
            intrusoAngulo = self.calcularAnguloIntruso(secondAcft, debug, xAtual, yAtual, vetorX, vetorY)
            matriz.append(intrusoAngulo)
            if debug:
                print("matriz depois de adicionar a tripla:")
                print(matriz)
            #remover a aeronave que já foi calculada do array
            index = [1]
            histConflitosLocal = np.delete(histConflitosLocal, index)

        if debug:
            print("matriz antes de adicionar os fakes:")
            print(matriz)

        # se for o caso, adicionar aeronaves até completar o número nnearest
        if len(histConflitos[current_timestep]) - 1 < nnearest:
            if len(histConflitos[current_timestep]) == 0:
                conflitosAadicionar = nnearest
            else:
                conflitosAadicionar = nnearest - (len(histConflitos[current_timestep])-1)
            if debug:
                print("vai adicionar %d conflitos fakes" %(conflitosAadicionar))
            for i in range(conflitosAadicionar):
                matriz.append(200) # distância fake
                matriz.append(0) # ângulos fake
                matriz.append(0)  # ângulos fake
        else:
            if debug:
                print("não vai adicionar conflitos fake porque len(histConflitos[current_timestep]) = %d" %(len(histConflitos[current_timestep])))

        return matriz

    def adicionaCandidatos (self, histConflitosLocal, debug, nnearest):
        if debug:
            print("função adiciona candidatos")
        candidatos = range(1, nnearest + 1, 1)
        foraDeConflito = [x for x in candidatos if x not in histConflitosLocal]
        if debug:
            print("candidatos:")
            print(candidatos)
            print("adicionando candidatos fora de conflito:")
            print(foraDeConflito)
        histConflitosLocal = np.concatenate((histConflitosLocal, foraDeConflito), axis=None)
        if debug:
            print("depois de adicionar:")
            print(histConflitosLocal)

    def calculaDataMatrix2(self, histConflitos, current_timestep, debug, posDRL, posicaoTempVoos, vetorX,
                                             vetorY, xAtual, yAtual, nnearest):
        matriz = []

        # criar cópia local para não alterar a lista original
        histConflitosLocal =  np.copy(histConflitos[current_timestep])

        if debug:
            print("histórico de conflitos original:")
            print(histConflitos[current_timestep])
            print("cópia local:")
            print(histConflitosLocal)

        #adicionar elementos à lista de conflitos para preencher os nnearest
        if histConflitosLocal == None:
            histConflitosLocal = []
            if debug:
                print("histórico de conflitos local estava vazio")
        else:
            if debug:
                print("histórico de conflitos local não estava vazio")

        #se for o caso, adicionar aeronaves até completar o número nnearest
        if histConflitosLocal == None :
            self.adicionaCandidatos(histConflitosLocal, debug, nnearest)
            if len(histConflitosLocal)-1 < nnearest:
                self.adicionaCandidatos(histConflitosLocal, debug, nnearest)
        else:
            if debug:
                print("naõ entrou no if porque len(histConflitosLocal)= %d" %(len(histConflitosLocal)))

        # preencher o data matrix com os dados das n aeronaves mais próximas
        for i in range(nnearest):
            if debug:
                print("loop do nnearest. i= %d. lista de conflitos: " % (i))
                print(histConflitosLocal)

            secondAcft = self.voos[0][int(histConflitosLocal[1])]
            if debug:
                print("vai calcular as distâncias para DRL")

            self.intrusoDist = int(self.calculaDistanciaPorEstrategia(histConflitosLocal, debug, posDRL,
                                                                      posicaoTempVoos))
            # tentativa de limitar a distância
            if self.intrusoDist > 200:
                self.intrusoDist = 200

            matriz.append(self.intrusoDist)
            matriz.append(posicaoTempVoos[secondAcft][0]+500)
            matriz.append(posicaoTempVoos[secondAcft][1]+500)
            matriz.append(self.vertis[0][secondAcft.destino][0]+500)
            matriz.append(self.vertis[0][secondAcft.destino][1]+500)





            if debug:
                print("matriz depois de adicionar as variáveis do novo intruso:")
                print(matriz)
            #remover a aeronave que já foi calculada do array
            index = [1]
            histConflitosLocal = np.delete(histConflitosLocal, index)



        return matriz

    def calculaNovoDesvio(self, debug, vooZero):
        xDestino = self.vertis[0][vooZero.destino][0]
        yDestino = self.vertis[0][vooZero.destino][1]

        vetorXDestino = xDestino - vooZero.xAtual
        vetorYDestino = yDestino - vooZero.yAtual
        if debug:
            print("vai calcular a proa do destino para redefinir o desvio")
        proaDestino = self.calcularAnguloIntrusoAux(vetorYDestino, vetorXDestino, debug, vooZero.vetorX,
                                                    vooZero.vetorY)
        if proaDestino > 27:
            novoDesvio = proaDestino - 27
        else:
            novoDesvio = proaDestino + 9

        if debug:
            print("novo desvio calculado: %d" % (novoDesvio))
        return novoDesvio

    def analisarDesvioDRL(self, debug, vooZero, proaRadianos, action):
        if debug:
            print("data matrix anterior completo:")
            print(self.dataMatrix)
        intrusoAngulo = self.dataMatrix[1]
        intrusoDist = self.dataMatrix[0]
        if debug:
            print("vai analisar o desvio para intruso angulo = %d" %(intrusoAngulo))
        if intrusoAngulo < 33 and intrusoAngulo > 3:
            # condição para determinar se o intruso está dentro do cone de 60 graus à frente
            if debug:
                print("intruso fora do cone")
            # avaliar a distância para decidir se irá executar o desvio

            if debug:
                print(" vai analisar a distância. intrusoDist = %f" %(intrusoDist))
            if intrusoDist > self.toleranciaLinear/2: # só vai ignorar o desvio se o intruso ainda estiver na zona de monitoramento
                # calcular o novo desvio
                if debug:
                    print("fora da zona de conflito. recalcular desvio")
                action['desvio'] = self.calculaNovoDesvio(debug, vooZero)

            else:
                if debug:
                    print("intruso está na zona de conflito. executar o desvio recomendado")

        else:
            if debug:
                print(" intruso dentro do cone de 60 graus. intrusoDist = %d" %(intrusoDist))
            if intrusoDist > self.toleranciaLinear *1.9 : # se a distância for muito grande, não é um conflito real
                if debug:
                    print("conflito fictício. recalcular desvio")
                action['desvio'] = self.calculaNovoDesvio(debug, vooZero)

            else: # o conflito é real, não alterar a action
                if debug:
                    print("o conflito é real. executará o desvio")

        return action




    def compute_timestep(self, action, current_timestep, debug, duracao, nnearest):

        """
        Compute the dynamics of the plane over a given number of episodes based on thrust and theta values
        Variables : Thrust in N, theta in degrees, number of episodes (no unit)
        This will be used by the RL environment.
        """

        self.timestep += 1 # increment timestep
        self.histConflitos[current_timestep] = [] #garrantir que o array esteja inicializado
        self.histConflitosAvoid[current_timestep] = [] #garrantir que o array esteja inicializado
        self.histConflitosFree[current_timestep] = [] #garrantir que o array esteja inicializado
        self.histConflitosRandom[current_timestep] = [] #garrantir que o array esteja inicializado
        if debug:
            print("******voo não radar. momento= %d. action recebida:" % (current_timestep))
            print(action)





        vooZero = self.voos[0][0]
        proaRadianos = math.atan2(vooZero.vetorY, vooZero.vetorX)

        # fazer com que o desvio seja executado como uma consequencia da action recebida
        # a action só será considerada se houver um conflito
        # colocar a condição abaixo no if para detectar conflitos desde o início
        # current_timestep > duracao / 10 and
        if self.histConflitos[current_timestep-1] is not None and len (self.histConflitos[current_timestep-1]) > 0 :
            if debug:
                print("vai analisar o desvio.")
            action = self.analisarDesvioDRL(debug, vooZero, proaRadianos, action)

            self.executarDesvioDRL(debug, vooZero, proaRadianos, action)
            # self.ajustarVelocidadeDRL(debug, vooZero, proaRadianos, action)
        else:
            if debug:
                print("não executou desvio. vai retornar à proa inicial")

            vetorX = self.vertis[0][vooZero.destino][0] - vooZero.xAtual
            vetorY = self.vertis[0][vooZero.destino][1] - vooZero.yAtual
            hipotenusa = math.sqrt(vetorX ** 2 + vetorY ** 2)
            vooZero.vetorX = vetorX/hipotenusa
            vooZero.vetorY = vetorY/hipotenusa

        # colocar a condição abaixo no if para detectar conflitos desde o início
        # current_timestep > duracao / 10 and
        if self.histConflitosAvoid[current_timestep - 1] is not None and len(self.histConflitosAvoid[current_timestep - 1]) > 0:
            secondAcft = self.histConflitosAvoid[current_timestep-1][1]
            self.executarDesvioAvoid(debug, secondAcft, vooZero)
        else:
            vetorX = self.vertis[0][vooZero.destino][0] - vooZero.xAtualAvoid
            vetorY = self.vertis[0][vooZero.destino][1] - vooZero.yAtualAvoid
            hipotenusa = math.sqrt(vetorX ** 2 + vetorY ** 2)
            vooZero.vetorXAvoid = vetorX/hipotenusa
            vooZero.vetorYAvoid = vetorY/hipotenusa

        # colocar a condição abaixo no if para detectar conflitos desde o início
        # current_timestep > duracao / 10 and
        if self.histConflitosRandom[current_timestep - 1] is not None and len(self.histConflitosRandom[current_timestep - 1]) > 0:
            self.executarDesvioRandom(vooZero, debug)
        else:
            vetorX = self.vertis[0][vooZero.destino][0] - vooZero.xAtualRandom
            vetorY = self.vertis[0][vooZero.destino][1] - vooZero.yAtualRandom
            hipotenusa = math.sqrt(vetorX ** 2 + vetorY ** 2)
            vooZero.vetorXRandom = vetorX/hipotenusa
            vooZero.vetorYRandom = vetorY/hipotenusa


        # para cada instante da simulação, movimenta todas as aeronaves em linha reta
        posicaoTempVoos = np.empty(self.quantAeronaves, dtype=object)
        for j in range(self.quantAeronaves):
            if debug:
                print ("vai chamar a função movimenta para voo: %d" %(j))
            resultMovimenta = self.movimenta(current_timestep, self.voos[0][j], self.vertis)
            if debug:
                print("result movimenta:")
                print(resultMovimenta)
            posicaoTempVoos[j] = resultMovimenta[0][0], resultMovimenta[0][1]
            if(j == 0):
                posDRL =    resultMovimenta[0][0], resultMovimenta[0][1]
                posAvoid =  resultMovimenta[1][0], resultMovimenta[1][1]
                posFree =   resultMovimenta[2][0], resultMovimenta[2][1]
                posRandom = resultMovimenta[3][0], resultMovimenta[3][1]

        self.historicoDeVoos[current_timestep] = posicaoTempVoos


        # detecção de conflitos apenas para a aeronave 0, pois ela será a única controlada pela IA
        if debug:
            print("vai fazer a detecção de conflitos")
            print(posicaoTempVoos)
        # buscar o conflito com a aeronave mais próxima

        # if current_timestep > duracao / 10:
        if debug:
            print("já passou dos primeiros 10% da duração. vai detectar conflitos")
        for k in range(1, self.quantAeronaves):
            if debug:
                print("variando o k na busca de conflitos. k atual = %d, " % (k))


            if debug:
                print(" vai buscar conflitos para DRL")
            # DRL detectar se a aeronave k é um intruso
            self.detectaConflitoPorEstrategia(self.histConflitos, current_timestep, posDRL,
                                              posicaoTempVoos, debug, k, nnearest)
            # AVOID
            if debug:
                print(" vai buscar conflitos para avoid")

            self.detectaConflitoPorEstrategia(self.histConflitosAvoid, current_timestep,
                                              posAvoid, posicaoTempVoos, debug, k, nnearest)
            # FREE
            if debug:
                print(" vai buscar conflitos para free")

            self.detectaConflitoPorEstrategia(self.histConflitosFree, current_timestep,
                                              posFree, posicaoTempVoos, debug, k, nnearest)
            # RANDOM
            if debug:
                print(" vai buscar conflitos para random")
            self.detectaConflitoPorEstrategia(self.histConflitosRandom, current_timestep,
                                              posRandom, posicaoTempVoos, debug, k, nnearest)
        # else:
        #     print("não vai detectar conflitos porque está dentro dos primeiros 10% de duração")


        # DRL
        # calcular a distância entre a aeronave zero e o intruso
        # if self.histConflitos[current_timestep] is not None:
        if debug:
            print ("vai calcular o data matrix para DRL")
        self.intrusoDistAnterior = self.intrusoDist #guardar o valor anterior antes de recalcular
        self.dataMatrixAnterior = self.dataMatrix[:]#guardar o valor anterior antes de recalcular
        if debug:
            print("data matrix anterior antes de calcular o novo valor:")
            print(self.dataMatrixAnterior)
        self.dataMatrix = self.calculaDataMatrix(self.histConflitos, current_timestep, debug, posDRL, posicaoTempVoos,
                                                     vooZero.vetorX, vooZero.vetorY, vooZero.xAtual, vooZero.yAtual, nnearest)
        # if debug:
        #     print("calculou o novo data matrix mas o anterior não deveria mudar.")
        #     print(self.dataMatrixAnterior)


        # else:
        #     #se não houver um conflito real, calcular as informações a partir da aeronave 1
        #     if debug:
        #         print ("não há conflito para DRL. vai passar as informações da aeronave 1")
        #     vooUm = self.voos[0][1]
        #     distUmX = vooUm.xAtual - vooZero.xAtual
        #     distUmY = vooUm.yAtual - vooZero.yAtual
        #     self.intrusoDistAnterior = self.intrusoDist  # guardar o valor anterior antes de recalcular
        #     self.intrusoDist = math.sqrt(distUmX**2 + distUmY**2)
        #     if debug:
        #         print("distância calculada para o voo 1: %f "%(self.intrusoDist))
        #     # angulo do intruso em relação à minha aeronave 12h = 0 graus, 1h = 30 graus
        #     self.intrusoAngulo = self.calcularAnguloIntruso(vooUm, debug, vooZero.xAtual,
        #                                                     vooZero.yAtual, vooZero.vetorX, vooZero.vetorY)
        #     self.anguloInterceptacao = 0


        #avoid
        # calcular a distância entre a aeronave zero e o intruso
        if len(self.histConflitosAvoid[current_timestep]) > 0:
            secondAcft = self.voos[0][self.histConflitosAvoid[current_timestep][1]]
            if debug:
                print ("vai calcular as distâncias para AVOID")
            distAvoid = self.calculaDistanciaPorEstrategia(self.histConflitosAvoid[current_timestep], debug,
                                                           posAvoid, posicaoTempVoos)
            if debug:
                print ("vai calcular a proa relativa para AVOID. o valor não será usado. apenas imprimir para debug")
                self.calculaAnguloInterceptacao(vooZero.vetorXAvoid, vooZero.vetorYAvoid, secondAcft, debug)
            if debug:
                print("vai calcular o ângulo do intruso para avoid")
            self.intrusoAnguloAvoid = self.calcularAnguloIntruso(secondAcft, debug, vooZero.xAtualAvoid,
                                                            vooZero.yAtualAvoid, vooZero.vetorXAvoid, vooZero.vetorYAvoid)
        else:
            if debug:
                print ("não há conflito para AVOID")
            distAvoid = 1000

        #FREE
        if len(self.histConflitosFree[current_timestep]) > 0:
            if debug:
                print("vai calcular a distância para free")
            distFree = self.calculaDistanciaPorEstrategia(self.histConflitosFree[current_timestep], debug,
                                                          posFree, posicaoTempVoos)
        else:
            if debug:
                print ("não há conflito para FREE")
            distFree = 1000

        #RANDOM
        # calcular a distância entre a aeronave zero e o intruso
        if len(self.histConflitosRandom[current_timestep]) > 0:
            if debug:
                print ("vai calcular as distâncias para RANDOM")
            distRandom = self.calculaDistanciaPorEstrategia(self.histConflitosRandom[current_timestep], debug,
                                                            posRandom, posicaoTempVoos)
        else:
            if debug:
                print ("não há conflito para RANDOM")
            distRandom = 1000

        arrayDist = []
        if current_timestep == duracao-1:
            # print("vai calcular as distâncias finais")
            vooZero = self.voos[0][0]
            #voo zero
            arrayDist.append(self.calculaDistPDestino (vooZero.xAtual, vooZero.yAtual, self.vertis[0][vooZero.destino][0], self.vertis[0][vooZero.destino][1], debug))
            # Avoid
            arrayDist.append(self.calculaDistPDestino(vooZero.xAtualAvoid, vooZero.yAtualAvoid,
                                self.vertis[0][vooZero.destino][0], self.vertis[0][vooZero.destino][1], debug))
            # Free
            arrayDist.append(self.calculaDistPDestino(vooZero.xAtualFree, vooZero.yAtualFree,
                                                      self.vertis[0][vooZero.destino][0],
                                                      self.vertis[0][vooZero.destino][1], debug))
            # Random
            arrayDist.append(self.calculaDistPDestino(vooZero.xAtualRandom, vooZero.yAtualRandom,
                                                      self.vertis[0][vooZero.destino][0],
                                                      self.vertis[0][vooZero.destino][1], debug))
            # print("dist[0] = %f" %(arrayDist[0]))







        reward, proaDestino = self.calculaReward (current_timestep, action['desvio'], debug, vooZero )

        # self.obs = [proaDestino] #proa do destino para comparar com a proa decidida pelo agente
        # self.obs.append(vooZero.xAtual+500) #x e y atuais são passados para comparação com a posição dos intrusos
        # self.obs.append(vooZero.yAtual+500) #o valor 500 é adicionado para criar um buffer e evitar números negativos

        # self.dataMatrix.append(vooZero.xAtual+500)
        # self.dataMatrix.append(vooZero.xAtual+500)
        # self.dataMatrix.append(self.vertis[0][vooZero.destino][0]+500)
        # self.dataMatrix.append(self.vertis[0][vooZero.destino][1]+500)
        self.obs = self.dataMatrix
        # self.obs = [int(self.intrusoDist), self.intrusoAngulo, self.anguloInterceptacao, proaDestino, 0]
        return self.obs, reward, distAvoid, distFree, distRandom, arrayDist

    



