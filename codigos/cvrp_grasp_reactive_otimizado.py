import argparse
import math
import random
import re


def _calcular_distancia_coordenadas(id_a, id_b, coordenadas, tipo_distancia):
    """Calcula uma distancia sem consultar a matriz pre-calculada."""
    x1, y1 = coordenadas[id_a]
    x2, y2 = coordenadas[id_b]
    distancia_euclidiana = math.hypot(x1 - x2, y1 - y2)

    if tipo_distancia == "EUC_2D":
        # Convencao EUC_2D da CVRPLIB: arredondamento para o inteiro mais proximo.
        return int(distancia_euclidiana + 0.5)

    if tipo_distancia in ("EXACT_2D", "EUC_2D_FLOAT"):
        return distancia_euclidiana

    raise ValueError(f"Tipo de distancia nao implementado: {tipo_distancia}")


def ler_instancia(caminho, numero_veiculos=None):
    """Le o arquivo .vrp no formato CVRPLIB e pre-calcula as distancias."""
    cabecalho = {}
    coordenadas = {}
    demandas = {}
    depositos = []
    secao = None

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue

            if linha == "EOF":
                break

            if linha in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"):
                secao = linha
                continue

            partes = linha.split()

            if secao == "NODE_COORD_SECTION":
                identificador = int(partes[0])
                x, y = float(partes[1]), float(partes[2])
                coordenadas[identificador] = (x, y)

            elif secao == "DEMAND_SECTION":
                identificador = int(partes[0])
                demandas[identificador] = int(partes[1])

            elif secao == "DEPOT_SECTION":
                identificador = int(partes[0])
                if identificador == -1:
                    secao = None
                else:
                    depositos.append(identificador)

            elif ":" in linha:
                chave, valor = linha.split(":", 1)
                cabecalho[chave.strip().upper()] = valor.strip()

    if not coordenadas or not demandas:
        raise ValueError("O arquivo nao contem as secoes de coordenadas e demandas.")
    if len(depositos) != 1:
        raise ValueError("A implementacao espera exatamente um deposito.")

    capacidade = int(cabecalho["CAPACITY"])
    tipo_distancia = cabecalho.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    nome = cabecalho.get("NAME", caminho)

    if numero_veiculos is None:
        # Em instancias CVRPLIB, o nome geralmente termina com -k seguido
        # do numero de veiculos, por exemplo: A-n33-k5.
        resultado = re.search(r"-k(\d+)", nome, re.IGNORECASE)
        if resultado:
            numero_veiculos = int(resultado.group(1))
        else:
            raise ValueError(
                "Nao foi possivel identificar o numero de veiculos no nome "
                "da instancia. Informe-o usando --veiculos."
            )

    deposito = depositos[0]
    clientes = []
    for identificador, coordenada in coordenadas.items():
        if identificador == deposito:
            continue
        if identificador not in demandas:
            raise ValueError(f"Demanda nao encontrada para o cliente {identificador}.")
        clientes.append(identificador)

    # A matriz evita repetir math.hypot e o arredondamento a cada consulta.
    # Dicionarios preservam a possibilidade de IDs nao consecutivos.
    matriz_distancias = {
        id_a: {
            id_b: _calcular_distancia_coordenadas(
                id_a, id_b, coordenadas, tipo_distancia
            )
            for id_b in coordenadas
        }
        for id_a in coordenadas
    }

    return {
        "nome": nome,
        "coordenadas": coordenadas,
        "demandas": demandas,
        "clientes": clientes,
        "deposito": deposito,
        "capacidade": capacidade,
        "veiculos": numero_veiculos,
        "tipo_distancia": tipo_distancia,
        "matriz_distancias": matriz_distancias,
    }


def distancia(id_a, id_b, instancia):
    """Consulta a distancia pre-calculada; mantem fallback para instancias antigas."""
    matriz = instancia.get("matriz_distancias")
    if matriz is not None:
        return matriz[id_a][id_b]
    return _calcular_distancia_coordenadas(
        id_a,
        id_b,
        instancia["coordenadas"],
        instancia["tipo_distancia"],
    )


def fitness_rota(rota, instancia):
    """Soma as distancias deposito -> clientes -> deposito."""
    if not rota:
        return 0

    deposito = instancia["deposito"]
    ponto_atual = deposito
    fitness = 0

    for cliente in rota:
        fitness += distancia(ponto_atual, cliente, instancia)
        ponto_atual = cliente

    # O retorno ao deposito faz parte da rota no problema CVRP.
    fitness += distancia(ponto_atual, deposito, instancia)
    return fitness


def fitness_total(solucao, instancia):
    return sum(fitness_rota(rota, instancia) for rota in solucao)


def construir_solucao_uma_vez(instancia, rng, alpha):
    """Construcao GRASP original: monta as rotas em ordem, respeitando Q."""
    clientes_restantes = instancia["clientes"][:]
    capacidade = instancia["capacidade"]
    deposito = instancia["deposito"]
    solucao = []

    for _ in range(instancia["veiculos"]):
        rota = []
        carga = 0
        ponto_atual = deposito

        while True:
            candidatos = [
                cliente
                for cliente in clientes_restantes
                if carga + instancia["demandas"][cliente] <= capacidade
            ]

            if not candidatos:
                break

            custos = {}
            distancia_atual_deposito = distancia(ponto_atual, deposito, instancia)
            for cliente in candidatos:
                custos[cliente] = (
                    distancia(ponto_atual, cliente, instancia)
                    + distancia(cliente, deposito, instancia)
                    - distancia_atual_deposito
                )

            menor_custo = min(custos.values())
            maior_custo = max(custos.values())
            limite = menor_custo + alpha * (maior_custo - menor_custo)
            rcl = [cliente for cliente in candidatos if custos[cliente] <= limite]
            escolhido = rng.choice(rcl)

            rota.append(escolhido)
            carga += instancia["demandas"][escolhido]
            ponto_atual = escolhido
            clientes_restantes.remove(escolhido)

        solucao.append(rota)
        if not clientes_restantes:
            break

    if clientes_restantes:
        return None

    while len(solucao) < instancia["veiculos"]:
        solucao.append([])

    return solucao


def construir_solucao(instancia, rng, alpha, max_tentativas=200):
    """Repete a construcao caso uma tentativa gere distribuicao inviavel."""
    for _ in range(max_tentativas):
        solucao = construir_solucao_uma_vez(instancia, rng, alpha)
        if solucao is not None:
            return solucao
    return None


def _fitness_com_uma_rota_alterada(custos_rotas, indice, novo_custo):
    """Recalcula o total com sum(), substituindo somente uma rota."""
    custos_candidatos = custos_rotas[:]
    custos_candidatos[indice] = novo_custo
    # Usar sum() preserva inclusive o comportamento de soma de floats
    # da versao do Python utilizada por fitness_total().
    return sum(custos_candidatos)


def _fitness_com_duas_rotas_alteradas(
    custos_rotas, indice_a, custo_a, indice_b, custo_b
):
    """Recalcula o total com sum(), substituindo somente duas rotas."""
    custos_candidatos = custos_rotas[:]
    custos_candidatos[indice_a] = custo_a
    custos_candidatos[indice_b] = custo_b
    return sum(custos_candidatos)


def _aplicar_movimento(solucao, movimento):
    """Aplica in-place o movimento que foi selecionado como melhor."""
    tipo = movimento[0]

    if tipo == "2-opt":
        _, rota, inicio, fim = movimento
        solucao[rota][inicio:fim + 1] = reversed(
            solucao[rota][inicio:fim + 1]
        )
    elif tipo == "relocate":
        _, origem, pos_origem, destino, pos_destino = movimento
        cliente = solucao[origem].pop(pos_origem)
        solucao[destino].insert(pos_destino, cliente)
    elif tipo == "swap":
        _, rota_a, pos_a, rota_b, pos_b = movimento
        solucao[rota_a][pos_a], solucao[rota_b][pos_b] = (
            solucao[rota_b][pos_b], solucao[rota_a][pos_a]
        )
    else:
        raise ValueError(f"Tipo de movimento desconhecido: {tipo}")


def busca_local_troca(solucao, instancia):
    """Busca local best-improvement com 2-opt, relocate e swap.

    Preserva a ordem das vizinhancas e dos candidatos da implementacao original.
    As candidatas copiam somente as rotas afetadas; o fitness das demais rotas
    e reutilizado. Ao final de cada varredura, somente o melhor movimento e
    aplicado. A busca aceita exclusivamente movimentos que reduzem o fitness.
    """
    atual = [rota[:] for rota in solucao]
    capacidade = instancia["capacidade"]
    demandas = instancia["demandas"]

    # Custos e cargas sao calculados uma vez por varredura e atualizados depois
    # que o melhor movimento da varredura e aplicado.
    custos_rotas = [fitness_rota(rota, instancia) for rota in atual]

    while True:
        fitness_atual = sum(custos_rotas)
        melhor_fitness = fitness_atual
        melhor_movimento = None
        cargas = [sum(demandas[c] for c in rota) for rota in atual]

        # 1. 2-opt: inverte segmentos dentro de cada rota.
        for i, rota in enumerate(atual):
            for inicio in range(len(rota) - 1):
                for fim in range(inicio + 1, len(rota)):
                    rota_candidata = rota[:]
                    rota_candidata[inicio:fim + 1] = reversed(
                        rota_candidata[inicio:fim + 1]
                    )
                    novo_custo = fitness_rota(rota_candidata, instancia)
                    valor = _fitness_com_uma_rota_alterada(
                        custos_rotas, i, novo_custo
                    )

                    if valor < melhor_fitness:
                        melhor_fitness = valor
                        melhor_movimento = ("2-opt", i, inicio, fim)

        # 2. Relocate: remove um cliente e o insere em outra posicao.
        # Inclui movimentos entre rotas e dentro da mesma rota.
        for origem in range(len(atual)):
            for pos_origem, cliente in enumerate(atual[origem]):
                for destino in range(len(atual)):
                    if origem != destino and (
                        cargas[destino] + demandas[cliente] > capacidade
                    ):
                        continue

                    for pos_destino in range(len(atual[destino]) + 1):
                        # Mantem a mesma regra de eliminacao de movimentos sem efeito.
                        if origem == destino and pos_destino in (
                            pos_origem, pos_origem + 1
                        ):
                            continue

                        if origem == destino:
                            rota_candidata = atual[origem][:]
                            movido = rota_candidata.pop(pos_origem)
                            rota_candidata.insert(pos_destino, movido)
                            novo_custo = fitness_rota(rota_candidata, instancia)
                            valor = _fitness_com_uma_rota_alterada(
                                custos_rotas, origem, novo_custo
                            )
                        else:
                            rota_origem = atual[origem][:]
                            rota_destino = atual[destino][:]
                            movido = rota_origem.pop(pos_origem)
                            rota_destino.insert(pos_destino, movido)
                            custo_origem = fitness_rota(rota_origem, instancia)
                            custo_destino = fitness_rota(rota_destino, instancia)
                            valor = _fitness_com_duas_rotas_alteradas(
                                custos_rotas,
                                origem,
                                custo_origem,
                                destino,
                                custo_destino,
                            )

                        if valor < melhor_fitness:
                            melhor_fitness = valor
                            melhor_movimento = (
                                "relocate",
                                origem,
                                pos_origem,
                                destino,
                                pos_destino,
                            )

        # 3. Swap: troca clientes entre duas rotas diferentes.
        for i in range(len(atual)):
            for j in range(i + 1, len(atual)):
                for pos_i, cliente_i in enumerate(atual[i]):
                    for pos_j, cliente_j in enumerate(atual[j]):
                        nova_carga_i = (
                            cargas[i] - demandas[cliente_i] + demandas[cliente_j]
                        )
                        nova_carga_j = (
                            cargas[j] - demandas[cliente_j] + demandas[cliente_i]
                        )
                        if nova_carga_i > capacidade or nova_carga_j > capacidade:
                            continue

                        rota_i = atual[i][:]
                        rota_j = atual[j][:]
                        rota_i[pos_i], rota_j[pos_j] = rota_j[pos_j], rota_i[pos_i]
                        custo_i = fitness_rota(rota_i, instancia)
                        custo_j = fitness_rota(rota_j, instancia)
                        valor = _fitness_com_duas_rotas_alteradas(
                            custos_rotas, i, custo_i, j, custo_j
                        )

                        if valor < melhor_fitness:
                            melhor_fitness = valor
                            melhor_movimento = ("swap", i, pos_i, j, pos_j)

        if melhor_movimento is None:
            break

        _aplicar_movimento(atual, melhor_movimento)

        # Atualiza os custos das rotas afetadas pelo movimento escolhido.
        tipo = melhor_movimento[0]
        if tipo == "2-opt":
            indice = melhor_movimento[1]
            custos_rotas[indice] = fitness_rota(atual[indice], instancia)
        elif tipo == "relocate":
            origem, destino = melhor_movimento[1], melhor_movimento[3]
            custos_rotas[origem] = fitness_rota(atual[origem], instancia)
            if destino != origem:
                custos_rotas[destino] = fitness_rota(atual[destino], instancia)
        elif tipo == "swap":
            rota_a, rota_b = melhor_movimento[1], melhor_movimento[3]
            custos_rotas[rota_a] = fitness_rota(atual[rota_a], instancia)
            custos_rotas[rota_b] = fitness_rota(atual[rota_b], instancia)

    return atual


def atualizar_probabilidades_reativas(
    alphas, somas_fitness, quantidades, melhor_fitness
):
    """Atualiza p(alpha) conforme Reactive GRASP: q_i=z*/A_i e p_i=q_i/soma(q)."""
    valores_q = []
    for indice in range(len(alphas)):
        if quantidades[indice] > 0:
            media = somas_fitness[indice] / quantidades[indice]
            valores_q.append(melhor_fitness / media if media > 0 else 1.0)
        else:
            # Mantem oportunidade de escolha para valores de alpha ainda nao testados.
            valores_q.append(1.0)

    soma_q = sum(valores_q)
    if soma_q <= 0:
        return [1.0 / len(alphas)] * len(alphas)
    return [q / soma_q for q in valores_q]


def grasp(instancia, iteracoes=50, alphas=None, semente=None, intervalo_reativo=1):
    """Reactive GRASP: alpha e escolhido adaptativamente; busca local ao final."""
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if not alphas or any(alpha < 0 or alpha > 1 for alpha in alphas):
        raise ValueError("Informe alphas entre 0 e 1.")
    if len(set(alphas)) != len(alphas):
        raise ValueError("A lista de alphas nao pode conter valores repetidos.")
    if intervalo_reativo <= 0:
        raise ValueError("O intervalo de atualizacao deve ser maior que zero.")
    if iteracoes < len(alphas):
        raise ValueError(
            f"O numero de iteracoes ({iteracoes}) deve ser pelo menos igual "
            f"a quantidade de alphas ({len(alphas)}), para que todos sejam avaliados."
        )

    rng = random.Random(semente)
    probabilidades = [1.0 / len(alphas)] * len(alphas)
    somas_fitness = [0.0] * len(alphas)
    quantidades = [0] * len(alphas)
    melhor_solucao = None
    melhor_fitness = float("inf")

    for iteracao in range(1, iteracoes + 1):
        # Fase inicial de exploracao: testa cada alpha pelo menos uma vez.
        # Depois disso, a escolha passa a seguir as probabilidades reativas.
        if iteracao <= len(alphas):
            indice_alpha = iteracao - 1
        else:
            indice_alpha = rng.choices(
                range(len(alphas)), weights=probabilidades, k=1
            )[0]
        alpha = alphas[indice_alpha]

        # Cada construcao recebe um gerador proprio. A semente principal
        # controla a sequencia de sementes e permite reproduzir a execucao.
        semente_construcao = rng.getrandbits(64)
        rng_construcao = random.Random(semente_construcao)
        solucao = construir_solucao(instancia, rng_construcao, alpha)
        if solucao is not None:
            # Nesta variacao, a busca local e aplicada depois da construcao completa.
            solucao = busca_local_troca(solucao, instancia)
            valor = fitness_total(solucao, instancia)
            somas_fitness[indice_alpha] += valor
            quantidades[indice_alpha] += 1

            if valor < melhor_fitness:
                melhor_fitness = valor
                melhor_solucao = [rota[:] for rota in solucao]

        if iteracao % intervalo_reativo == 0 and melhor_solucao is not None:
            probabilidades = atualizar_probabilidades_reativas(
                alphas, somas_fitness, quantidades, melhor_fitness
            )

    if melhor_solucao is None:
        raise ValueError(
            "Nao foi encontrada uma distribuicao valida. Verifique as demandas, "
            "a capacidade total e tente outra semente."
        )

    probabilidades = atualizar_probabilidades_reativas(
        alphas, somas_fitness, quantidades, melhor_fitness
    )
    estatisticas = []
    for indice, alpha in enumerate(alphas):
        media = (
            somas_fitness[indice] / quantidades[indice]
            if quantidades[indice] else None
        )
        estatisticas.append({
            "alpha": alpha,
            "probabilidade": probabilidades[indice],
            "quantidade": quantidades[indice],
            "media_fitness": media,
        })

    return melhor_solucao, melhor_fitness, estatisticas


def validar_solucao(solucao, instancia):
    """Confere se todos os clientes aparecem uma vez e se as cargas sao validas."""
    todos_os_clientes = [cliente for rota in solucao for cliente in rota]
    clientes_esperados = set(instancia["clientes"])

    if len(todos_os_clientes) != len(set(todos_os_clientes)):
        raise ValueError("Solucao invalida: um cliente foi atendido mais de uma vez.")
    if set(todos_os_clientes) != clientes_esperados:
        raise ValueError("Solucao invalida: ha clientes faltando ou nao reconhecidos.")

    for numero, rota in enumerate(solucao, start=1):
        carga = sum(instancia["demandas"][c] for c in rota)
        if carga > instancia["capacidade"]:
            raise ValueError(f"Solucao invalida: veiculo {numero} excedeu a capacidade.")


def main():
    parser = argparse.ArgumentParser(
        description="Resolucao de CVRP usando Reactive GRASP com 2-opt, relocate e swap (otimizado)."
    )
    parser.add_argument("arquivo", help="Caminho para a instancia .vrp")
    parser.add_argument("--iteracoes", type=int, default=50, help="Numero de iteracoes GRASP")
    parser.add_argument(
        "--alphas",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Valores candidatos de alpha separados por virgula",
    )
    parser.add_argument(
        "--intervalo-reativo",
        type=int,
        default=1,
        help="Iteracoes entre atualizacoes das probabilidades",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente aleatoria opcional; omita para gerar uma nova a cada execucao",
    )
    parser.add_argument(
        "--veiculos",
        type=int,
        default=None,
        help="Numero de veiculos, caso nao esteja no nome da instancia",
    )
    args = parser.parse_args()

    # Sem --seed, cada execucao do programa recebe uma semente principal nova.
    if args.seed is None:
        args.seed = random.SystemRandom().getrandbits(64)

    if args.iteracoes <= 0:
        parser.error("--iteracoes deve ser maior que zero.")
    if args.intervalo_reativo <= 0:
        parser.error("--intervalo-reativo deve ser maior que zero.")
    try:
        alphas = [float(v.strip()) for v in args.alphas.split(",") if v.strip()]
    except ValueError:
        parser.error("--alphas deve conter numeros separados por virgula.")
    if not alphas or any(a < 0 or a > 1 for a in alphas):
        parser.error("Informe ao menos um alpha, todos entre 0 e 1.")
    if len(set(alphas)) != len(alphas):
        parser.error("--alphas nao pode conter valores repetidos.")

    instancia = ler_instancia(args.arquivo, args.veiculos)
    if any(
        instancia["demandas"][c] > instancia["capacidade"]
        for c in instancia["clientes"]
    ):
        raise ValueError("Existe cliente com demanda maior que a capacidade de um veiculo.")
    demanda_total = sum(
        instancia["demandas"][c] for c in instancia["clientes"]
    )
    if demanda_total > instancia["capacidade"] * instancia["veiculos"]:
        raise ValueError("A demanda total ultrapassa a capacidade total da frota.")

    solucao, valor, estatisticas = grasp(
        instancia, args.iteracoes, alphas, args.seed, args.intervalo_reativo
    )
    validar_solucao(solucao, instancia)

    print(f"Instancia: {instancia['nome']}")
    print(
        f"Veiculos: {instancia['veiculos']} | "
        f"Capacidade individual: {instancia['capacidade']}"
    )
    print(f"Iteracoes: {args.iteracoes} | Seed: {args.seed}")
    print(f"Atualizacao reativa a cada {args.intervalo_reativo} iteracoes")
    print("\nEstatisticas finais dos alphas:")
    for item in estatisticas:
        if item["media_fitness"] is None:
            media = "sem solucao valida"
        else:
            media = f"{item['media_fitness']:.2f}"

        print(
            f"alpha={item['alpha']:.2f} | "
            f"probabilidade final={item['probabilidade']:.4f} | "
            f"solucoes validas={item['quantidade']} | "
            f"fitness medio={media}"
        )

    print("\nRotas encontradas:")
    for numero, rota in enumerate(solucao, start=1):
        carga = sum(instancia["demandas"][c] for c in rota)
        caminho = [instancia["deposito"]] + rota + [instancia["deposito"]]
        print(
            f"Veiculo {numero}: {' -> '.join(map(str, caminho))} "
            f"| demanda: {carga}/{instancia['capacidade']} "
            f"| distancia: {fitness_rota(rota, instancia)}"
        )
    print(f"\nDistancia total (fitness): {valor}")


if __name__ == "__main__":
    main()
