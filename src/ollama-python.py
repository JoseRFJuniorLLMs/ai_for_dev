import json
import ollama
import asyncio


# Simula uma chamada de API para obter os horários de voo
# Em uma aplicação real, isso buscaria dados de um banco de dados ou API ao vivo
def get_flight_times(departure: str, arrival: str) -> str:
    flights = {
        'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
        'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
        'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
        'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
        'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
        'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
    }

    key = f'{departure}-{arrival}'.upper()
    return json.dumps(flights.get(key, {'error': 'Voo não encontrado'}))


async def run(model: str):
    try:
        print("Inicializando o cliente Ollama...")
        client = ollama.AsyncClient(timeout=30.0)

        print("Enviando consulta inicial para o modelo...")
        messages = [{'role': 'user', 'content': 'Qual é o horário do voo de Nova York (NYC) para Los Angeles (LAX)?'}]

        print("Aguardando resposta do modelo...")
        response = await client.chat(
            model=model,
            messages=messages,
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_flight_times',
                        'description': 'Obter os horários de voo entre duas cidades',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'departure': {
                                    'type': 'string',
                                    'description': 'Cidade de partida (código do aeroporto)',
                                },
                                'arrival': {
                                    'type': 'string',
                                    'description': 'Cidade de chegada (código do aeroporto)',
                                },
                            },
                            'required': ['departure', 'arrival'],
                        },
                    },
                },
            ],
        )
        print("Recebida a resposta inicial do modelo.")
        print("Detalhes da resposta:", response)

        messages.append(response['message'])

        if not response['message'].get('tool_calls'):
            print("O modelo não usou a função. A resposta foi:")
            print(response['message']['content'])
            return

        print("O modelo decidiu usar a função fornecida.")
        if response['message'].get('tool_calls'):
            available_functions = {
                'get_flight_times': get_flight_times,
            }
            for tool in response['message']['tool_calls']:
                function_name = tool['function']['name']
                if function_name not in available_functions:
                    print(f"A função {function_name} não está disponível.")
                    continue

                function_to_call = available_functions[function_name]
                function_args = json.loads(tool['function']['arguments'])
                print(f"Chamando a função {function_name} com os argumentos: {function_args}")
                function_response = function_to_call(
                    function_args['departure'],
                    function_args['arrival']
                )
                print(f"Resposta da função: {function_response}")
                messages.append(
                    {
                        'role': 'tool',
                        'content': function_response,
                    }
                )

        print("Enviando consulta final para o modelo...")
        final_response = await client.chat(model=model, messages=messages)
        print("Resposta final do modelo:")
        print(final_response['message']['content'])

    except ollama.ResponseError as e:
        print(f"Erro da API do Ollama: {e}")
    except asyncio.TimeoutError:
        print("Conexão excedeu o tempo limite. Verifique se o Ollama está em execução e acessível.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


# Verifica se o serviço Ollama está em execução
async def check_ollama_service():
    try:
        print("Verificando o serviço Ollama...")
        client = ollama.AsyncClient(timeout=5.0)
        # Usa uma simples solicitação de lista de modelos para verificar se o serviço está responsivo
        await client.list()
        print("O serviço Ollama está em execução e acessível.")
        return True
    except Exception as e:
        print(f"Falha na verificação do serviço Ollama: {e}")
        print("Por favor, certifique-se de que o Ollama está instalado e em execução.")
        return False


# Executa as funções assíncronas
async def main():
    if await check_ollama_service():
        print("Iniciando execução principal...")
        await run('llama3.1:latest')
    else:
        print("Saindo devido à falha na verificação do serviço Ollama.")


if __name__ == "__main__":
    asyncio.run(main())
