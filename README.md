# DASF Seismic Lite

Este repositório contém uma implementação de referência para classificação de fácies sísmicas com o K-means usando o DASF. O *pipeline* de classificação é mostrado na figura abaixo [![pipeline](./pipeline.svg)](./pipeline.svg)

Para permitir a extração de atributos sísmicos, uma versão leve do `dasf-seismic`, intitulada `dasf-seismic-lite`, é disponibilizada no diretório `dasf-seismic-lite`.

## Instalação

Para instalar o `dasf-seismic-lite`:

1. Clone o repositório do [`dasf-core`](https://github.com/discovery-unicamp/dasf-core) e crie uma imagem docker de cpu. Para isso, você pode executar os seguintes comandos **(em uma pasta fora deste repositório)**, que irá gerar uma imagem docker chamada `dasf:cpu`:

```bash
git clone https://github.com/discovery-unicamp/dasf-core.git
cd dasf-core/build
CONTAINER_CMD=docker ./build_container.sh cpu
```

2. Clone este repositório e crie uma imagem docker que irá instalar o `dasf-seismic-lite`. Neste repositório, foi disponibilizado um *script* para facilitar a criação da imagem docker. Para isso, entre dentro do diretório raiz deste repositório e execute o *script* `build_docker.sh`, que irá gerar uma imagem docker chamada `dasf-seismic:cpu`, conforme abaixo:

```bash
./build_docker.sh
```

## Execução único nó

1. Dentro da pasta `data` deste repositótio, você deve colocar seus arquivos sísmicos `npy` de treino disponibilizados. No caso, o arquivo [`F3_train.npy`](https://drive.google.com/file/d/1N7r-6BsTW8HasiFVEzGnin2LxrRPfkjd/view?usp=drive_web&authuser=1).

2. Execute o comando abaixo para executar o *script* `reference.py` dentro do container. O parâmetro `--data` informa o caminho do dado numpy:
    
```bash
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 reference.py --data data/F3_train.npy
```


## Execução multi-nós

1. Instancie um *dask scheduler* (o endereço do *scheduler* será mostrado no terminal, será algo semelhante a `tcp://192.168.1.164:8786`):

```bash
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu dask scheduler
```

2. Instancie um *dask worker* e conecte-o ao *shceduler* criado no passo anterior (substitua `<scheduler_address>` pelo endereço do *scheduler*, será algo semelhante a `tcp://192.168.1.164:8786`)

```bash
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu dask worker <scheduler_address>
```

3. Execute a implementação de referencia (`reference.py`) passando o endereço do *scheduler* para o parâmetro `--address` (substitua `<scheduler_address>` pelo endereço do *scheduler*, será algo semelhante a `tcp://192.168.1.164:8786`)

```bash
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 reference.py --data data/F3_train.npy --address <scheduler_address>
```

**NOTA**: Lembre-se de executar os comandos acima na pasta raiz deste repositório.

## Informações úteis

* Voce pode acessar o dashborard do dask no nó que executa o *scheduler*, através do endereço http://localhost:8787/status
* Você pode salvar o resultado da predição da inline 0, feito pelo K-Means adicionando o parâmetro `--save-inline-fig output.png` ao *script* que `reference.py`. 
